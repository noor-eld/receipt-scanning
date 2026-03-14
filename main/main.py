# main.py
# FastAPI server: receives receipt images, runs OCR, and classifies items
# using a trained ML model (TF-IDF + Logistic Regression).
#
# HOW TO RUN:
#   1. First train the model: python train_model.py
#   2. Then start the server: python main.py
#   3. Test at http://localhost:8000/docs

import base64
import os

import requests as http_requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from classifier import classify_item, should_skip_line

load_dotenv()

GOOGLE_CLOUD_VISION_KEY = os.getenv("GOOGLE_CLOUD_VISION_KEY")
VISION_API_URL = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_CLOUD_VISION_KEY}"

app = FastAPI(
    title="Recipe App - Receipt Scanner API",
    description="Processes receipt images and extracts food ingredients using ML",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    Processes raw OCR text into classified items using the ML model.

    Steps:
    1. Split into lines
    2. Skip receipt metadata (dates, totals, store info)
    3. Classify each line with the ML model
    4. Return food items and low-confidence items for user review
    """

    lines = ocr_text.strip().split("\n")
    results = []
    seen = set()

    for line in lines:
        line = line.strip()

        if not line or should_skip_line(line):
            continue

        # Use the ML model to classify
        prediction = classify_item(line)

        label = prediction["label"]
        confidence = prediction["confidence"]

        # Skip items the model is confident are non-food
        if label == "non_food" and confidence > 0.7:
            continue

        # Avoid duplicates
        if line.lower() in seen:
            continue
        seen.add(line.lower())

        results.append({
            "raw_text": line,
            "category": label,
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
    Receives a receipt image, runs OCR, classifies items with ML model,
    and returns detected food items with confidence scores.
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
        "message": f"Found {len(ingredients)} item(s)",
        "ingredients": ingredients,
        "raw_text": ocr_text,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)