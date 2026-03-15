# normalizer.py
# Normalizes messy receipt text into clean ingredient names using NLP techniques.
#
# APPROACH: Fuzzy String Matching
# Instead of hardcoding "pnt = peanut" and "brst = breast:"
#   1. Clean the text (remove barcodes, prices, weights — basic preprocessing)
#   2. Compare the cleaned text against a DATABASE of known ingredient names
#   3. Use fuzzy matching to find the closest match
#
# The library "rapidfuzz" implements fuzzy matching using algorithms like
# Levenshtein distance (counting how many letter changes turn one string into another)
# and token-based matching (comparing word by word regardless of order).

import re
from rapidfuzz import fuzz, process

# --- INGREDIENT DATABASE ---
# This is the "dictionary" of clean ingredient names.
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

    Process:
    1. Preprocess: remove barcodes, prices, weights, codes
    2. Fuzzy match against ingredient database
    3. If good match found (score >= 60), return the clean database name
    4. If no match, return the preprocessed text as-is (user can edit it)

    The score threshold of 60 was chosen because:
    - Exact matches score 100 ("broccoli" → "broccoli")
    - Close matches score 70-90 ("brocoli" → "broccoli")
    - Partial matches score 50-70 ("GV PNT BUTTR" → "peanut butter")
    - Unrelated text scores below 50

    fuzz.token_set_ratio is used as the scorer because it:
    - Ignores word order ("green grapes" matches "grapes green")
    - Handles extra words ("organic banana loose" still matches "banana")
    - Compares meaningful tokens, not just raw characters
    """

    cleaned = _preprocess_text(raw_text)

    if not cleaned:
        return raw_text.lower().strip()

    # process.extractOne compares cleaned text against every entry
    # in INGREDIENT_DATABASE and returns the best match with a score.
    result = process.extractOne(
        query=cleaned,
        choices=INGREDIENT_DATABASE,
        scorer=fuzz.token_set_ratio,
        score_cutoff=60.0,
    )

    if result:
        match, score, index = result
        return match
    else:
        # No good match — return cleaned text for the user to review
        return cleaned