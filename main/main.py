# main.py
# A FastAPI server that receives receipt images, extracts text via OCR,
# and classifies items as food or non-food.
#
# WHAT THIS DOES:
# 1. Receives an uploaded receipt image
# 2. Sends it to Google Cloud Vision for OCR
# 3. Splits the text into lines and classifies each as food / non-food / unknown
# 4. Returns only food and unknown items (non-food is filtered out)
#
# HOW TO RUN:
#   python main.py
#   Then visit http://localhost:8000/docs to test it

import base64
import os

import requests as http_requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from classifier import classify_item, should_skip_line

# Load environment variables from .env file
load_dotenv()

GOOGLE_CLOUD_VISION_KEY = os.getenv("GOOGLE_CLOUD_VISION_KEY")
VISION_API_URL = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_CLOUD_VISION_KEY}"

# --- Create the app ---
app = FastAPI(
    title="Recipe App - Receipt Scanner API",
    description="Processes receipt images and extracts food ingredients",
    version="0.2.0",
)

# CORS middleware — allows the phone app to talk to this server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Sends an image to Google Cloud Vision API and returns all detected text.
    """

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
    Takes raw OCR text and returns a list of classified items.

    Steps:
    1. Split text into lines
    2. Skip non-product lines (dates, totals, store info)
    3. Classify each remaining line as food / non_food / unknown
    4. Return food and unknown items (user will review unknowns)
    """

    lines = ocr_text.strip().split("\n")
    results = []
    seen = set()  # Track duplicates

    for line in lines:
        line = line.strip()

        if not line or should_skip_line(line):
            continue

        category = classify_item(line)

        # Skip non-food items
        if category == "non_food":
            continue

        # Skip duplicates
        if line.lower() in seen:
            continue
        seen.add(line.lower())

        results.append({
            "raw_text": line,
            "category": category,
        })

    return results


# --- ENDPOINTS ---

@app.get("/health")
async def health_check():
    """Simple check to verify the server is running."""
    return {"status": "ok"}


@app.post("/api/receipt/process")
async def process_receipt(image_file: UploadFile = File(...)):
    """
    Receives a receipt image, runs OCR, classifies items as food/non-food,
    and returns the detected food items.
    """

    # Read the uploaded image
    try:
        image_bytes = await image_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")

    # Run OCR
    ocr_text = extract_text_from_image(image_bytes)

    if not ocr_text:
        return {
            "success": False,
            "message": "Could not read text from the image. Try a clearer photo.",
            "ingredients": [],
            "raw_text": "",
        }

    # Classify items
    ingredients = parse_receipt_text(ocr_text)

    return {
        "success": True,
        "message": f"Found {len(ingredients)} item(s)",
        "ingredients": ingredients,
        "raw_text": ocr_text,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)