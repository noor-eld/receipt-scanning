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
    "good-quality", "sturdy", "qt", "shallow", "ceramic", "glass",
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


# --- Quick test: run this file directly to verify it loads correctly ---
# Usage: cd into main/ folder, then run: python recipe_recommender.py
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: CSV not found at {DATA_PATH}")
    else:
        print("Loading recipe database...")
        df = pd.read_csv(DATA_PATH)
        print(f"Total rows in CSV: {len(df)}")

        # Test parsing on first 5 recipes
        print("\n--- Sample parsed ingredients ---")
        for i in range(5):
            title = df.iloc[i]["Title"]
            raw = df.iloc[i]["Ingredients"]
            cleaned = _parse_ingredients_list(str(raw))
            print(f"\n{title}:")
            print(f"  Parsed: {cleaned}")