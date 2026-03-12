# main.py
# A FastAPI server that receives receipt images and extracts text using Google Cloud Vision.
#
# WHAT THIS DOES:
# 1. Receives an uploaded image from the phone app
# 2. Sends it to Google Cloud Vision for OCR (text reading)
# 3. Returns the raw text back
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

# Load environment variables from .env file
load_dotenv()

GOOGLE_CLOUD_VISION_KEY = os.getenv("GOOGLE_CLOUD_VISION_KEY")
VISION_API_URL = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_CLOUD_VISION_KEY}"

# --- Create the app ---
app = FastAPI(
    title="Recipe App - Receipt Scanner API",
    description="Processes receipt images and extracts text",
    version="0.1.0",
)

# CORS middleware — allows the phone app to talk to this server.
# Without this, requests from a different device/origin get blocked.
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

    How it works:
    1. Convert the image to base64 (a text format Google's API accepts)
    2. Send it to the Vision API with a TEXT_DETECTION request
    3. Pull the full text out of Google's response
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


# --- ENDPOINTS ---

@app.get("/health")
async def health_check():
    """Simple check to verify the server is running."""
    return {"status": "ok"}


@app.post("/api/receipt/process")
async def process_receipt(image_file: UploadFile = File(...)):
    """
    Receives a receipt image, runs OCR, and returns the extracted text.

    For now, this returns raw text only.
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
            "raw_text": "",
        }

    return {
        "success": True,
        "message": "Text extracted successfully",
        "raw_text": ocr_text,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)