# recipe_recommender.py
# Loads the recipe database and parses ingredient lists.
#
# The CSV stores ingredients as Python list strings:
#   "['2 Tbsp. butter', '1 cup flour', ...]"
#
# We need to:
# 1. Parse that string into an actual Python list
# 2. Strip quantities, units, and prep words to get base ingredient names
#    e.g., "2 Tbsp. finely chopped fresh rosemary" → "rosemary"

import os
import re
import ast
import pandas as pd
from rapidfuzz import fuzz

# --- Path to the recipe CSV ---
# os.path.dirname(__file__) gets the folder this script is in (main/)
# Then we go up one level (..) into "data/"
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "13k-recipes.csv"
)

# --- Words to strip from ingredient strings ---
# These are NOT ingredients — they're measurements and cooking instructions.

UNITS = {
    "teaspoon", "teaspoons", "tsp", "tablespoon", "tablespoons", "tbsp",
    "cup", "cups", "ounce", "ounces", "oz", "pound", "pounds", "lb", "lbs",
    "gram", "grams", "kg", "ml", "liter", "liters", "quart", "quarts",
    "pint", "pints", "gallon", "gallons", "inch", "inches", "can", "cans",
    "jar", "jars", "package", "packages", "bunch", "bunches", "head",
    "heads", "clove", "cloves", "slice", "slices", "piece", "pieces",
    "stick", "sticks", "sprig", "sprigs", "pinch", "dash", "handful",
}

PREP_WORDS = {
    "chopped", "minced", "diced", "sliced", "crushed", "grated", "shredded",
    "peeled", "seeded", "cored", "halved", "quartered", "cubed", "julienned",
    "torn", "trimmed", "melted", "softened", "sifted", "toasted", "ground",
    "freshly", "finely", "coarsely", "thinly", "roughly", "lightly",
    "divided", "optional", "packed", "slivered", "blanched", "dried",
    "frozen", "thawed", "room", "temperature", "unsalted", "salted",
    "extra-virgin", "low-sodium", "whole", "large", "medium", "small",
    "fresh", "plus", "more", "for", "serving", "garnish", "to", "taste",
    "about", "approximately", "loosely", "firmly", "into", "cut", "removed", "or", "and", "of", "with", "such", "as",
    "good-quality", "sturdy", "qt", "shallow", "ceramic", "glass", "cold", "reserved",
    "another", "use",
}


def _clean_ingredient(raw: str) -> str:
    """
    Strips quantities, units, and prep words from a raw ingredient string.

    "2 Tbsp. finely chopped fresh rosemary" → "rosemary"
    "1 (3½–4-lb.) whole chicken"            → "chicken"
    """

    text = raw.lower().strip()

    # Remove unicode fractions (½, ¼, ¾, etc.)
    text = re.sub(r"[\u00BC-\u00BE\u2150-\u215E\u2189]", "", text)

    # Remove numbers (including decimals like 3.5)
    text = re.sub(r"\d+\.?\d*", "", text)

    # Remove parenthetical info like "(about 3 lb. total)"
    text = re.sub(r"\(.*?\)", "", text)

    # Remove punctuation except hyphens
    text = re.sub(r"[,;.!?\"\'""'']", "", text)

    # Split into words and filter out units/prep words
    words = text.split()
    filtered = []

    for word in words:
        word = word.strip("-– ")

        if not word or len(word) < 2:
            continue
        if word in UNITS or word in PREP_WORDS:
            continue

        filtered.append(word)

    return " ".join(filtered).strip()


def _parse_ingredients_list(ingredients_str: str) -> list[str]:
    """
    Parses the Ingredients column from the CSV.

    The CSV stores ingredients as a string that looks like a Python list:
    "['2 Tbsp. butter', '1 cup flour', ...]"

    ast.literal_eval safely converts this string into an actual list.
    Then we clean each ingredient to get just the base name.
    """

    try:
        raw_list = ast.literal_eval(ingredients_str)
    except (ValueError, SyntaxError):
        return []

    cleaned = []
    for item in raw_list:
        clean = _clean_ingredient(item)
        if clean and len(clean) > 1:
            cleaned.append(clean)

    return cleaned

def _match_ingredient(pantry_item: str, recipe_ingredient: str) -> bool:
    """
    Checks if a pantry item matches a recipe ingredient.

    Uses TWO strategies:
    1. Substring: "chicken" matches "chicken breast"
    2. Fuzzy: "parmesan" matches "parmigiano" (75% threshold)

    WHY BOTH?
    Substring catches cases where pantry has "chicken" and the recipe
    says "chicken breast" — fuzzy might score this low because "breast"
    adds noise. Fuzzy catches misspellings and variations that substring misses.
    """

    pantry_lower = pantry_item.lower().strip()
    recipe_lower = recipe_ingredient.lower().strip()

    # Strategy 1: Direct substring match
    if pantry_lower in recipe_lower or recipe_lower in pantry_lower:
        return True

    # Strategy 2: Fuzzy match (sorts words before comparing,
    # so "red onion" matches "onion red")
    score = fuzz.token_sort_ratio(pantry_lower, recipe_lower)
    return score >= 75

class RecipeRecommender:
    """
    Loads recipes from CSV and recommends based on available ingredients.

    WHY A CLASS?
    It loads 13k recipes into memory ONCE when the server starts.
    A class holds this data so it doesn't reload the CSV on every request.
    The server creates one instance, and every API call reuses the same data.
    """

    def __init__(self):
        self.recipes = []
        self._load_recipes()

    def _load_recipes(self):
        """Load and preprocess all recipes from CSV."""

        if not os.path.exists(DATA_PATH):
            print(f"WARNING: Recipe data not found at {DATA_PATH}")
            return

        print("Loading recipe database...")
        df = pd.read_csv(DATA_PATH)

        count = 0
        for _, row in df.iterrows():
            title = row.get("Title", "")
            ingredients_raw = row.get("Ingredients", "[]")
            instructions = row.get("Instructions", "")

            if not title or not ingredients_raw:
                continue

            cleaned_ingredients = _parse_ingredients_list(str(ingredients_raw))

            # Skip recipes with fewer than 2 ingredients (probably bad data)
            if len(cleaned_ingredients) < 2:
                continue

            self.recipes.append({
                "title": str(title),
                "ingredients": cleaned_ingredients,
                "ingredient_count": len(cleaned_ingredients),
                "instructions": str(instructions),
            })
            count += 1

        print(f"Loaded {count} recipes.")

    def recommend(
            self,
            pantry_items: list[str],
            top_n: int = 10,
            min_match_percent: float = 0.3,
    ) -> list[dict]:
        """
        Scores all recipes against the user's pantry.

        For each recipe: score = (ingredients you HAVE) / (total ingredients)
        Returns the top matches sorted by score.

        Args:
            pantry_items: e.g., ["chicken", "butter", "garlic", "salt"]
            top_n: How many results to return
            min_match_percent: Minimum score to include (0.3 = 30%)
        """

        if not pantry_items or not self.recipes:
            return []

        scored = []

        for recipe in self.recipes:
            matched = []
            missing = []

            for recipe_ing in recipe["ingredients"]:
                found = False

                for pantry_item in pantry_items:
                    if _match_ingredient(pantry_item, recipe_ing):
                        matched.append(recipe_ing)
                        found = True
                        break

                if not found:
                    missing.append(recipe_ing)

            total = len(recipe["ingredients"])
            match_score = len(matched) / total if total > 0 else 0

            if match_score >= min_match_percent and recipe["ingredient_count"] >= 4:
                scored.append({
                    "title": recipe["title"],
                    "match_score": round(match_score, 2),
                    "matched_ingredients": matched,
                    "missing_ingredients": missing,
                    "total_ingredients": total,
                    "instructions": recipe["instructions"],
                })

        # Sort by match score first, then by number of matched ingredients
        # This prevents tiny 2-ingredient recipes from always ranking first.
        # A 70% match on 10 ingredients (7 matches) is more useful
        # than 100% on 2 ingredients (2 matches).
        scored.sort(
        key=lambda x: (x["match_score"], len(x["matched_ingredients"])),
        reverse=True,
        )

        return scored[:top_n]

    def get_all_recipes(
            self,
            page: int = 1,
            per_page: int = 20,
    ) -> dict:
        """
        Returns paginated recipes for browsing (no pantry matching).

        WHY PAGINATION?
        13k recipes is too many to send at once — it would freeze the app.
        Pagination sends 20 at a time. The app loads more as the user scrolls.
        """

        start = (page - 1) * per_page
        end = start + per_page

        page_recipes = self.recipes[start:end]

        return {
            "recipes": [
                {
                    "title": r["title"],
                    "ingredient_count": r["ingredient_count"],
                    "ingredients": r["ingredients"],
                    "instructions": r["instructions"],
                }
                for r in page_recipes
            ],
            "total": len(self.recipes),
            "page": page,
            "per_page": per_page,
            "total_pages": (len(self.recipes) + per_page - 1) // per_page,
        }

    def get_recipe_by_title(self, title: str) -> dict | None:
        """Returns full recipe details for a single recipe."""

        for recipe in self.recipes:
            if recipe["title"].lower() == title.lower():
                return recipe
        return None


# --- Quick test: run this file directly to verify it loads correctly ---
# Usage: cd into main/ folder, then run: python recipe_recommender.py
if __name__ == "__main__":
    recommender = RecipeRecommender()

    # Test with some sample pantry items
    test_pantry = ["chicken", "butter", "garlic", "salt", "pepper", "rosemary"]
    print(f"\nPantry: {test_pantry}")
    print("--- Top 3 Recommendations ---")

    results = recommender.recommend(test_pantry, top_n=3)
    for r in results:
        print(f"\n  {r['title']}")
        print(f"  Match: {int(r['match_score'] * 100)}%")
        print(f"  Have: {r['matched_ingredients']}")
        print(f"  Need: {r['missing_ingredients']}")

    # Test browsing
    print("\n--- Browse page 1 ---")
    page = recommender.get_all_recipes(page=1, per_page=3)
    print(f"Total recipes: {page['total']}")
    for r in page["recipes"]:
        print(f"  {r['title']} ({r['ingredient_count']} ingredients)")