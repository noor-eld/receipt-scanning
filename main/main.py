import cv2
import numpy as np
import pytesseract
import re

# Set Tesseract path (macOS Homebrew)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# -------- STEP 1: Load Image --------
image1 = cv2.imread('../data/receipt1.jpeg')
image3 = cv2.imread('../data/receipt3.jpg')


if image1 is None:
    raise FileNotFoundError("Could not read ../data/receipt1.jpeg")

original = image1.copy()

# -------- STEP 2: Preprocess --------
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# edged = cv2.Canny(blur, 75, 200)

# -------- STEP 3: Simple Crop --------
warped = original.copy()

# -------- STEP 4: OCR Enhancement --------
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# Increase contrast slightly
warped_gray = cv2.convertScaleAbs(warped_gray, alpha=1.5, beta=0)

# Otsu threshold
_, thresh = cv2.threshold(
    warped_gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# -------- STEP 5: OCR --------
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(thresh, config=custom_config)

print("\n====== OCR OUTPUT ======\n")
print(text)

# -------- STEP 6: Structured Extraction --------

# Extract total
total_match = re.search(r'TOTAL\.?\s*\$?(\d+\.\d{2})', text)
total = total_match.group(1) if total_match else "Not found"

# Extract date
date_match = re.search(r'\d{2}/\d{2}/\d{4}', text)
date = date_match.group(0) if date_match else "Not found"

print("\n====== STRUCTURED DATA ======")
print("Date:", date)
print("Total:", total)

# -------- DISPLAY --------
cv2.imshow("Original", original)
cv2.imshow("Warped", warped)
cv2.imshow("Threshold", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()