# main.py
# FastAPI server: receives receipt images, runs OCR, classifies items with ML,
# and normalizes food items into clean ingredient names.
#
# HOW TO RUN:
#   1. Train the model: python train_model.py
#   2. Start the server: python main.py
#   3. Test at http://localhost:8000/docs

import os

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from classifier import classify_item, should_skip_line
from normalizer import normalize_ingredient

from recipe_recommender import RecipeRecommender

import cv2
import numpy as np
import pytesseract

load_dotenv()

# Tesseract path (macOS Homebrew)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

app = FastAPI(
    title="Recipe App - Receipt Scanner API",
    description="Processes receipt images and extracts food ingredients using ML",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load recipe database into memory once at startup
recommender = RecipeRecommender()


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Processes a receipt image using OpenCV preprocessing and Tesseract OCR.

    Steps:
    1. Convert raw bytes to an OpenCV image
    2. Convert to grayscale (removes colour noise)
    3. Increase contrast (makes text stand out from background)
    4. Apply Gaussian blur (reduces small noise/grain)
    5. Apply Otsu thresholding (converts to pure black and white)
    6. Run Tesseract OCR on the cleaned image

    """

    # Convert raw bytes into a numpy array, then into an OpenCV image
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        print("Failed to decode image")
        return ""

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Increase contrast — makes text darker, background lighter
    # alpha=1.5 boosts contrast, beta=0 keeps brightness neutral
    contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

    # Step 3: Gaussian blur — smooths out small noise/grain
    blur = cv2.GaussianBlur(contrast, (5, 5), 0)

    # Step 4: Otsu thresholding — automatically finds the best cutoff
    # to convert the image to pure black and white
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 5: Run Tesseract OCR
    # --oem 3: use the best available OCR engine (LSTM neural net)
    # --psm 6: assume the image is a single block of text (good for receipts)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    return text

def parse_receipt_text(ocr_text: str) -> list[dict]:
    """
    Full receipt processing pipeline:
    1. Split OCR text into lines
    2. Skip receipt metadata (preprocessing)
    3. Classify each line with ML model
    4. Normalize food items into clean ingredient names
    5. Return only food items with clean names
    """

    lines = ocr_text.strip().split("\n")
    results = []
    seen = set()

    for line in lines:
        line = line.strip()

        if not line or should_skip_line(line):
            continue

        # ML classification
        prediction = classify_item(line)
        label = prediction["label"]
        confidence = prediction["confidence"]

        if label != "food":
            continue

        # Normalize: "GV PNT BUTTR 007874237003 F" → "peanut butter"
        ingredient = normalize_ingredient(line)

        # Skip if normalization returned empty or duplicate
        if not ingredient or ingredient in seen:
            continue
        seen.add(ingredient)

        results.append({
            "raw_text": line,
            "ingredient": ingredient,
            "confidence": round(confidence, 2),
        })

    return results


# --- ENDPOINTS ---

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/receipt/process")
async def process_receipt(image_file: UploadFile = File(...)):
    """
    Receives a receipt image, runs OCR, classifies items with ML,
    normalizes ingredient names, and returns clean results.
    """

    try:
        image_bytes = await image_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")

    ocr_text = extract_text_from_image(image_bytes)

    if not ocr_text:
        return {
            "success": False,
            "message": "Could not read text from the image. Try a clearer photo.",
            "ingredients": [],
            "raw_text": "",
        }

    ingredients = parse_receipt_text(ocr_text)

    return {
        "success": True,
        "message": f"Found {len(ingredients)} ingredient(s)",
        "ingredients": ingredients,
        "raw_text": ocr_text,
    }

@app.post("/api/recipes/recommend")
async def recommend_recipes(pantry_items: list[str], top_n: int = 10):
    """
    Accepts the user's pantry ingredients and returns the best recipe matches.
    Expects a JSON body: ["chicken", "butter", "garlic", "salt"]
    """

    if not pantry_items:
        return {"success": False, "message": "No pantry items provided", "recipes": []}

    results = recommender.recommend(pantry_items, top_n=top_n)

    return {
        "success": True,
        "message": f"Found {len(results)} matching recipe(s)",
        "recipes": results,
    }


@app.get("/api/recipes/browse")
async def browse_recipes(page: int = 1, per_page: int = 20):
    """
    Returns paginated recipes for browsing without matching.
    """

    result = recommender.get_all_recipes(page=page, per_page=per_page)

    return {
        "success": True,
        **result,
    }

@app.get("/api/recipes/detail")
async def recipe_detail(title: str):
    """Returns full details for a single recipe by title."""

    recipe = recommender.get_recipe_by_title(title)

    if not recipe:
        return {"success": False, "message": "Recipe not found"}

    return {
        "success": True,
        "recipe": recipe,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)