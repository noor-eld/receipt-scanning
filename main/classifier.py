# classifier.py
# ONE JOB: Look at a receipt line and decide — is this food or not?
#
# WHY KEYWORDS INSTEAD OF ML?
# A TF-IDF + Logistic Regression model needs thousands
# of labeled training examples and can still get confused by abbreviations like
# "GV PNT BUTTR". A keyword-based approach is more reliable for receipt text,
# easier to debug, and easier to extend.

import re

# --- KNOWN FOOD WORDS ---
# If any of these appear in a receipt line, it's probably food.
FOOD_KEYWORDS = {
    # Proteins
    "chicken", "beef", "pork", "turkey", "salmon", "tuna", "fish", "shrimp",
    "prawn", "lamb", "steak", "mince", "sausage", "bacon", "ham", "chkn",
    "brst", "breast", "thigh", "wing", "drumstick", "fillet", "bnls",

    # Dairy
    "milk", "cheese", "butter", "yogurt", "yoghurt", "cream", "egg", "eggs",
    "cheddar", "mozzarella", "parmesan", "parm", "buttr",

    # Fruits
    "apple", "banana", "orange", "lemon", "lime", "grape", "grapes",
    "strawberry", "strawberries", "blueberry", "blueberries", "mango",
    "pear", "peach", "melon", "watermelon", "pineapple", "avocado",
    "cherry", "cherries", "raspberry", "raspberries", "kiwi",

    # Vegetables
    "potato", "potatoes", "tomato", "tomatoes", "onion", "onions",
    "garlic", "carrot", "carrots", "broccoli", "broc", "spinach",
    "lettuce", "pepper", "peppers", "cucumber", "celery", "corn",
    "mushroom", "mushrooms", "zucchini", "courgette", "cabbage",
    "cauliflower", "bean", "beans", "pea", "peas", "sprouts",
    "asparagus", "kale", "leek",

    # Grains & Bread
    "bread", "rice", "pasta", "noodle", "noodles", "flour", "oats",
    "cereal", "tortilla", "wrap", "bagel", "croissant", "fusilli",
    "spaghetti", "penne", "macaroni", "mahatma", "jasm", "jasmine",

    # Pantry staples
    "sugar", "salt", "oil", "olive", "vinegar", "sauce", "ketchup",
    "mustard", "mayo", "mayonnaise", "honey", "jam", "peanut",
    "sriracha", "hoisin", "soy", "oyster", "sesame", "pepper",
    "pepercrn", "peppercorn", "cumin", "paprika", "cinnamon",
    "oregano", "basil", "thyme", "rosemary",

    # Snacks & Baking
    "chocolate", "chips", "crisps", "cookie", "cookies", "biscuit",
    "cracker", "crackers", "popcorn", "nuts", "almond", "almonds",
    "walnut", "cashew", "muffin", "muffins", "cake",

    # Beverages
    "juice", "coffee", "tea", "water", "soda", "folgers",

    # Canned / Preserved
    "soup", "broth", "stock", "canned", "frozen", "organic",

    # Common receipt abbreviations
    "pnt", "chkn", "chnk", "grn", "brn", "whl",
}

# --- KNOWN NON-FOOD WORDS ---
NON_FOOD_KEYWORDS = {
    # Cleaning
    "soap", "detergent", "bleach", "cleaner", "cleaning", "wipe", "wipes",
    "sponge", "trash", "garbage", "tissue", "tissues",
    "towel", "towels", "toilet", "dishwash",

    # Personal care
    "shampoo", "conditioner", "toothpaste", "toothbrush", "deodorant",
    "lotion", "razor", "shaving", "cosmetic", "lipstick",
    "makeup", "nail", "cotton", "bandage", "medicine",

    # Household
    "battery", "batteries", "bulb", "candle", "foil", "plastic",
    "container", "hanger", "tape", "glue", "pen", "pencil", "marker",
    "nitril", "nitrile", "glove", "gloves",

    # Store-related lines to ignore
    "subtotal", "total", "tax", "change", "cash", "credit", "debit",
    "visa", "mastercard", "amex", "tend", "balance", "savings",
    "coupon", "reward", "discount", "loyalty", "member", "receipt",
    "thank", "thanks", "welcome", "store", "manager", "phone",
    "refund", "return", "exchange", "items sold",
}

# --- RECEIPT LINE PATTERNS TO SKIP ---
# Entire lines that are clearly not product lines.
SKIP_PATTERNS = [
    r"^\s*$",                          # Empty lines
    r"^\d{2}/\d{2}/\d{2,4}",          # Dates (06/01/2016)
    r"^\d{2}:\d{2}",                   # Times (08:26)
    r"^ST#\s*\d+",                     # Store numbers
    r"^TE#\s*\d+",                     # Terminal numbers
    r"^TR#\s*\d+",                     # Transaction numbers
    r"^\s*\d+\.\d{2}\s*(lb|kg)",       # Weight-only lines
    r"^VAT",                           # VAT lines
    r"^EFT",                           # Electronic fund transfer
    r"^PAYMENT",                       # Payment lines
    r"^ACCOUNT",                       # Account lines
    r"^REF\s*#",                       # Reference numbers
    r"^NETWORK",                       # Network ID lines
    r"^TC#",                           # Transaction codes
    r"^\s*\*+\s*$",                    # Just asterisks
    r"^-+$",                           # Just dashes
    r"^\d{4}\s+\d+",                   # Barcode-only lines
    r"^Enter survey",                  # Survey prompts
    r"^Layaway",                       # Store promotions
    r"^Save money",                    # Store slogans
    r"^You saved",                     # Savings lines
]


def should_skip_line(line: str) -> bool:
    """Check if a line should be completely ignored (dates, totals, store info, etc.)."""
    line = line.strip()
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False


def classify_item(item_text: str) -> str:
    """
    Takes a receipt item line and returns "food", "non_food", or "unknown".

    How it works:
    1. Extract just the words (ignore numbers and symbols)
    2. Count how many words match food keywords vs non-food keywords
    3. Non-food takes priority — better to miss a food item than include soap

    Examples:
        "GV PNT BUTTR 007874237003 F"  → "food"
        "12 CT NITRIL"                  → "non_food"
        "SRIRACHA SCE"                  → "food"
    """
    lower_text = item_text.lower()
    words = re.findall(r"[a-z]+", lower_text)

    food_matches = sum(1 for word in words if word in FOOD_KEYWORDS)
    non_food_matches = sum(1 for word in words if word in NON_FOOD_KEYWORDS)

    if non_food_matches > 0 and non_food_matches >= food_matches:
        return "non_food"

    if food_matches > 0:
        return "food"

    return "unknown"