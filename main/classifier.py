# classifier.py
# Classifies receipt items as food or non-food using a trained ML model.
#
# This uses a TF-IDF + Logistic Regression pipeline trained on labeled grocery data.
# The model was trained by running train_model.py, which saved it to food_classifier.pkl.
#
# The model still uses regex patterns to skip non-product lines (dates, totals, etc.)
# — that's preprocessing, not classification. The ML model handles the actual
# food vs non-food decision.

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


# --- RECEIPT LINE PATTERNS TO SKIP ---
# These are not product lines — they're receipt metadata.
# This is preprocessing.
SKIP_PATTERNS = [
    r"^\s*$",                          # Empty lines
    r"^\d{2}/\d{2}/\d{2,4}",          # Dates
    r"^\d{2}:\d{2}",                   # Times
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
    r"^SUBTOTAL",                      # Subtotal lines
    r"^TOTAL",                         # Total lines
    r"^TAX",                           # Tax lines
    r"^CHANGE",                        # Change lines
    r"^CASH",                          # Cash lines
    r"^DEBIT",                         # Debit lines
    r"^CREDIT",                        # Credit lines
    r"^TEND",                          # Tender lines
    r"^AMEX",                          # Payment method
    r"^VISA",                          # Payment method
    r"^ITEMS SOLD",                    # Items count
    r"^Coupon",                        # Coupon lines
    r"^LOYALTY",                       # Loyalty lines
    r"^SALES",                         # Sales lines
    r"^SPECIAL$",                      # Special lines with no product name
]


def should_skip_line(line: str) -> bool:
    """Check if a line is receipt metadata (not a product)."""
    line = line.strip()
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False


def classify_item(item_text: str) -> dict:
    """
    Uses the trained ML model to classify a receipt line.

    Returns a dict with:
    - label: "food" or "non_food"
    - confidence: how sure the model is (0.0 to 1.0)

    If the model isn't loaded, defaults to "unknown".
    """
    if model is None:
        return {"label": "unknown", "confidence": 0.0}

    prediction = model.predict([item_text])[0]
    probabilities = model.predict_proba([item_text])[0]
    confidence = float(max(probabilities))

    return {
        "label": prediction,
        "confidence": confidence,
    }