# normalizer.py
# Normalizes messy receipt text into clean ingredient names using NLP techniques.
#
# APPROACH: Fuzzy String Matching
# Instead of hardcoding "pnt = peanut" and "brst = breast", we:
#   1. Clean the text (remove barcodes, prices, weights — basic preprocessing)
#   2. Compare the cleaned text against a DATABASE of known ingredient names
#   3. Use fuzzy matching to find the closest match
#
# WHY THIS IS BETTER THAN HARDCODING:
#   - Handles misspellings we've never seen (e.g., "BROCOLI" → "broccoli")
#   - Handles abbreviations without explicit mappings (e.g., "CHKN BRST" → "chicken breast")
#   - Adapts to new receipts — if the ingredient is in the database, it gets matched
#   - Adding new ingredients = adding to the database, not writing new code
#
# The library "rapidfuzz" implements fuzzy matching using algorithms like
# Levenshtein distance (counting how many letter changes turn one string into another)
# and token-based matching (comparing word by word regardless of order).

import re
from rapidfuzz import fuzz, process

# --- INGREDIENT DATABASE ---
# This is our "dictionary" of clean ingredient names.
# Fuzzy matching compares receipt text against these to find the best match.
# To support a new ingredient, just add it here — no code changes needed.
INGREDIENT_DATABASE = [
    # Fruits
    "apple", "apricot", "avocado", "banana", "blackberry", "blueberry",
    "cherry", "clementine", "coconut", "cranberry", "fig", "grape",
    "grapefruit", "kiwi", "lemon", "lime", "mango", "melon", "nectarine",
    "orange", "passion fruit", "peach", "pear", "pineapple", "plum",
    "pomegranate", "raspberry", "satsuma", "strawberry", "watermelon",

    # Vegetables
    "asparagus", "aubergine", "beetroot", "broccoli", "brussels sprouts",
    "butternut squash", "cabbage", "carrot", "cauliflower", "celery",
    "courgette", "zucchini", "corn", "sweet corn", "cucumber", "green beans",
    "kale", "leek", "lettuce", "mushroom", "onion", "red onion", "spring onion",
    "parsnip", "peas", "snow peas", "sugar snap peas", "pepper", "bell pepper",
    "chili pepper", "potato", "sweet potato", "pumpkin", "radish", "rocket",
    "spinach", "squash", "tomato", "cherry tomato", "turnip", "watercress",
    "baby corn", "mange tout", "pak choi", "fennel", "artichoke",

    # Herbs
    "basil", "bay leaves", "chives", "coriander", "dill", "mint",
    "oregano", "parsley", "rosemary", "sage", "thyme", "tarragon",

    # Proteins
    "chicken breast", "chicken thigh", "chicken drumstick", "chicken wing",
    "whole chicken", "chicken", "beef", "beef mince", "steak", "sirloin",
    "pork", "pork chop", "pork loin", "pork belly", "bacon", "ham",
    "lamb", "lamb chop", "lamb mince", "turkey", "turkey breast",
    "salmon", "tuna", "cod", "haddock", "prawns", "shrimp", "sea bass",
    "mackerel", "sardines", "fish fingers", "sausage", "chorizo",
    "salami", "pepperoni", "prosciutto",

    # Dairy & Eggs
    "milk", "whole milk", "semi skimmed milk", "skimmed milk", "oat milk",
    "almond milk", "soy milk", "butter", "cheese", "cheddar", "mozzarella",
    "parmesan", "feta", "brie", "gouda", "halloumi", "cream cheese",
    "cottage cheese", "ricotta", "eggs", "yogurt", "greek yogurt",
    "double cream", "single cream", "sour cream", "creme fraiche",

    # Grains & Bread
    "bread", "white bread", "wholemeal bread", "sourdough", "bagel",
    "croissant", "pitta bread", "tortilla wrap", "naan bread", "brioche",
    "ciabatta", "baguette", "bread roll", "rice", "basmati rice",
    "brown rice", "jasmine rice", "long grain rice", "wild rice",
    "pasta", "penne", "fusilli", "spaghetti", "tagliatelle", "linguine",
    "macaroni", "lasagne sheets", "noodles", "egg noodles", "rice noodles",
    "udon noodles", "flour", "self raising flour", "cornflour",
    "oats", "porridge oats", "cereal", "granola", "muesli",
    "couscous", "quinoa", "bulgur wheat",

    # Pantry staples
    "olive oil", "vegetable oil", "sunflower oil", "coconut oil",
    "sesame oil", "avocado oil", "rapeseed oil", "vinegar",
    "balsamic vinegar", "apple cider vinegar", "sugar", "brown sugar",
    "caster sugar", "icing sugar", "salt", "sea salt", "honey",
    "maple syrup", "golden syrup",

    # Sauces & Condiments
    "soy sauce", "oyster sauce", "hoisin sauce", "sriracha",
    "tomato ketchup", "mustard", "mayonnaise", "bbq sauce", "hot sauce",
    "worcestershire sauce", "pesto", "sweet chili sauce",
    "peanut butter", "nutella", "jam", "marmalade",

    # Spices
    "black pepper", "peppercorn", "cumin", "paprika", "smoked paprika",
    "cinnamon", "turmeric", "chili powder", "curry powder", "garam masala",
    "nutmeg", "mixed herbs", "italian seasoning", "chinese five spice",

    # Stock & Canned
    "chicken stock", "beef stock", "vegetable stock", "stock cube",
    "tomato puree", "chopped tomatoes", "passata", "coconut milk",
    "coconut cream", "baked beans", "kidney beans", "chickpeas",
    "lentils", "red lentils", "black beans", "butter beans",
    "canned tuna", "soup",

    # Snacks & Baking
    "chocolate", "dark chocolate", "milk chocolate", "crisps", "chips",
    "tortilla chips", "popcorn", "almonds", "cashews", "walnuts",
    "pistachios", "mixed nuts", "sesame seeds", "pumpkin seeds",
    "sunflower seeds", "chia seeds", "flax seeds", "raisins",
    "biscuits", "crackers", "rice cakes", "muffin", "cake",

    # Beverages
    "orange juice", "apple juice", "coffee", "instant coffee",
    "coffee beans", "tea bags", "green tea", "herbal tea",

    # Other
    "hummus", "guacamole", "salsa", "coleslaw", "tofu", "tempeh",
    "protein bar", "ice cream", "frozen pizza", "frozen peas",
    "chicken nuggets", "fish fingers",
]


def _preprocess_text(raw_text: str) -> str:
    """
    Basic text cleanup BEFORE fuzzy matching.

    This removes things that are clearly not part of the ingredient name
    so fuzzy matching works better. This is preprocessing, not normalization.
    """

    text = raw_text.lower().strip()

    # Remove barcodes (6+ digit sequences)
    text = re.sub(r"\d{6,}", "", text)

    # Remove prices ($4.66, £0.79)
    text = re.sub(r"[\$£€]?\s*-?\d+\.\d{2}", "", text)

    # Remove trailing store codes (single letter: F, N, T, X, A, B, O)
    text = re.sub(r"\s+[fntkxabo]\s*$", "", text)

    # Remove weight/size info (1L, 500g, 16oz, 1kg, 12ct)
    text = re.sub(r"\d+\s*(l|ml|g|kg|oz|lb|ct|pk|pt)\b", "", text, flags=re.IGNORECASE)

    # Remove quantity lines (0.778kg NET @ $5.99/kg)
    text = re.sub(
        r"\d+\.?\d*\s*(kg|lb|g)\s*(net)?\s*(@|at)?\s*[\$£€]?\d*\.?\d*\s*/?\s*(kg|lb|g)?",
        "", text, flags=re.IGNORECASE
    )

    # Remove "2 x £0.31" style text
    text = re.sub(r"\d+\s*x\s*[\$£€]?\d+\.?\d*", "", text, flags=re.IGNORECASE)

    # Remove remaining standalone numbers
    text = re.sub(r"\b\d+\b", "", text)

    # Clean up extra whitespace
    text = " ".join(text.split()).strip()

    return text


def normalize_ingredient(raw_text: str) -> str:
    """
    Takes messy receipt text and returns a clean ingredient name.

    Uses a MULTI-STRATEGY approach to find the best match:

    Strategy 1: Match the FULL cleaned text against the database.
        Good for: "BANANA CAVENDISH" → "banana" (full phrase comparison)

    Strategy 2: Match EACH WORD individually against the database.
        Good for: "ZUCHINNI GREEN" → "zucchini" (the important word is zuchinni,
        not green — matching word by word catches this)
    """

    cleaned = _preprocess_text(raw_text)

    if not cleaned:
        return raw_text.lower().strip()

    best_match = None
    best_score = 0.0

    # Strategy 1: Match the full cleaned text
    # Uses WRatio which tries multiple comparison methods and picks the best.
    # Good for multi-word ingredients like "peanut butter" or "cherry tomato"
    full_result = process.extractOne(
        query=cleaned,
        choices=INGREDIENT_DATABASE,
        scorer=fuzz.WRatio,
        score_cutoff=60.0,
    )

    if full_result:
        match, score, index = full_result
        if score > best_score:
            best_match = match
            best_score = score

    # Strategy 2: Match each word individually
    # Split "ZUCHINNI GREEN" into ["zuchinni", "green"] and match each one.
    # This catches cases where one word is the ingredient and the other is noise.
    words = cleaned.split()
    for word in words:
        if len(word) < 3:  # Skip tiny words like "gv", "sc"
            continue

        word_result = process.extractOne(
            query=word,
            choices=INGREDIENT_DATABASE,
            scorer=fuzz.ratio,  # Strict character comparison for single words
            score_cutoff=70.0,  # Higher threshold for single words to avoid false matches
        )

        if word_result:
            match, score, index = word_result
            # Boost single-word matches slightly less than full-phrase matches
            # so "peanut butter" (full match) beats "peanut" (word match)
            adjusted_score = score * 0.95
            if adjusted_score > best_score:
                best_match = match
                best_score = adjusted_score

    if best_match:
        return best_match
    else:
        return cleaned