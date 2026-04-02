# main.py
# FastAPI server: receives receipt images, runs OCR, classifies items with ML,
# and normalizes food items into clean ingredient names.
#
# HOW TO RUN:
#   1. Train the model: python train_model.py
#   2. Start the server: python main.py
#   3. Test at http://localhost:8000/docs

import base64
import os

import requests as http_requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from classifier import classify_item, should_skip_line
from normalizer import normalize_ingredient

from recipe_recommender import RecipeRecommender

load_dotenv()

GOOGLE_CLOUD_VISION_KEY = os.getenv("GOOGLE_CLOUD_VISION_KEY")
VISION_API_URL = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_CLOUD_VISION_KEY}"

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
    """Sends an image to Google Cloud Vision API and returns all detected text."""

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "requests": [
            {
                "image": {"content": base64_image},
                "features": [{"type": "TEXT_DETECTION"}],
            }
        ]
    }

    response = http_requests.post(VISION_API_URL, json=payload)

    if response.status_code != 200:
        print(f"Vision API error: {response.status_code} - {response.text}")
        return ""

    result = response.json()

    try:
        annotations = result["responses"][0].get("textAnnotations", [])
        if annotations:
            return annotations[0]["description"]
        return ""
    except (KeyError, IndexError) as e:
        print(f"Error parsing Vision API response: {e}")
        return ""


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