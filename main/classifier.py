# classifier.py
# Classifies receipt items as food or non-food using a trained ML model.
#
# Two layers of filtering:
# 1. PREPROCESSING (regex) — catches obvious non-product lines like prices,
#    dates, weights, and store info BEFORE they reach the model
# 2. ML MODEL — classifies remaining lines as food or non_food
#
# Why both? The ML model was trained on product names like "Chicken Breast"
# and "Toilet Cleaner". It was never trained on "$4.66" or "0.778kg NET",
# so it guesses poorly on those. Preprocessing removes them so the model
# only sees actual product lines.

import re
import pickle
import os

# --- Load the trained ML model ---
MODEL_PATH = "food_classifier.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"ML model loaded from {MODEL_PATH}")
else:
    model = None
    print(f"WARNING: {MODEL_PATH} not found. Run train_model.py first.")


# --- Minimum confidence to accept a prediction ---
# If the model is less than 70% sure, the item is marked "unknown"
# so the user can decide. This prevents junk lines from slipping through.
CONFIDENCE_THRESHOLD = 0.70


# --- RECEIPT LINE PATTERNS TO SKIP ---
# These are receipt metadata, not product lines.
# This is preprocessing, not classification.
SKIP_PATTERNS = [
    # Empty and whitespace
    r"^\s*$",
    r"^\s*\*+\s*$",                    # Just asterisks
    r"^-+$",                           # Just dashes

    # Prices — lines that are ONLY a price (with or without currency symbol)
    r"^[\$£€]?\s*-?\d+\.\d{2}\s*$",

    # Weight/quantity lines (0.778kg NET @ $5.99/kg)
    r"^\d+\.?\d*\s*kg\s+NET",
    r"^\d+\.?\d*\s*lb\s+",

    # Dates and times
    r"^\d{2}/\d{2}/\d{2,4}",
    r"^\d{2}:\d{2}",

    # Store metadata
    r"^ST#\s*\d+",
    r"^TE#\s*\d+",
    r"^TR#\s*\d+",
    r"^OP#\s*\d+",

    # Financial lines
    r"^SUBTOTAL",
    r"^TOTAL",
    r"^TAX",
    r"^CHANGE",
    r"^CASH",
    r"^DEBIT",
    r"^CREDIT",
    r"^TEND",
    r"^AMEX",
    r"^VISA",
    r"^MASTERCARD",
    r"^BALANCE",
    r"^VAT",
    r"^EFT",
    r"^PAYMENT",
    r"^ACCOUNT",

    # Store info
    r"^REF\s*#",
    r"^NETWORK",
    r"^TC#",
    r"^\d{4}\s+\d+",                   # Barcode-only lines
    r"^Enter survey",
    r"^Layaway",
    r"^Save money",
    r"^You saved",
    r"^ITEMS SOLD",
    r"^Coupon",
    r"^LOYALTY",
    r"^SALES",
    r"^SPECIAL$",

    # Very short lines — single words that are never product names
    # (3 or fewer characters with no real product info)
    r"^[A-Z]{1,3}$",                   # "WED", "MON", "TAX", etc.
    r"^DATE$",
]


def should_skip_line(line: str) -> bool:
    """
    Check if a line is receipt metadata that should be filtered out
    before reaching the ML model.
    """
    line = line.strip()

    for pattern in SKIP_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True

    # Additional check: if a line contains NO letters, skip it.
    # Product names always have at least some text.
    # This catches stray numbers, prices, and codes.
    if not re.search(r"[a-zA-Z]", line):
        return True

    return False


def classify_item(item_text: str) -> dict:
    """
    Uses the trained ML model to classify a receipt line.

    Returns a dict with:
    - label: "food", "non_food", or "unknown"
    - confidence: how sure the model is (0.0 to 1.0)

    If confidence is below the threshold, label becomes "unknown"
    so the user can review it manually.
    """
    if model is None:
        return {"label": "unknown", "confidence": 0.0}

    prediction = model.predict([item_text])[0]
    probabilities = model.predict_proba([item_text])[0]
    confidence = float(max(probabilities))

    # If the model isn't confident enough, mark as unknown
    if confidence < CONFIDENCE_THRESHOLD:
        return {"label": "unknown", "confidence": confidence}

    return {
        "label": prediction,
        "confidence": confidence,
    }