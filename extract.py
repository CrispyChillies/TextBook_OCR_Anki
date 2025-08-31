import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import numpy as np
import cv2
import pytesseract
import json
from pprint import pprint

config = "--oem 1 --psm 7 -l vie"

data = []
pdf_path = "Module GPDC-2.pdf"
pages = convert_from_path(pdf_path)


for i, page in enumerate(pages):
    page = np.array(page)

    if i == 0:
        header_height = 300
        footer_height = 300
    else:
        header_height = 175
        footer_height = 250

    page = page[header_height:-footer_height, :, :]

    # Grey formatting
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Dilation formatting
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Identify contours and y coordinates
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 29 and w > 60:
            roi = page[y : y + h, x : x + w]
            blocks.append((y, roi))

    # Sort the content following the y coordinates
    blocks = sorted(blocks, key=lambda b: b[0])

    # Extract text
    for _, roi in blocks:
        roi = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
        text = pytesseract.image_to_string(roi, config=config)
        text = text.replace("\n", " ")
        text = text.replace("ŒC.", "C.")
        text = text.replace("Œ.", "C.")
        data.append(text)


with open("data.txt", "w") as file:
    for line in data:
        file.write(line + "\n")

questions = []

i = 0
while i < len(data):
    question_text = data[i].strip()
    if i + 4 <= len(data):
        options = data[i + 1 : i + 5]
        struct = {
            "Question": question_text,
            "A": options[0].strip(),
            "B": options[1].strip(),
            "C": options[2].strip(),
            "D": options[3].strip(),
        }
        questions.append(struct)
        i += 5
    else:
        break

with open("questions.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print("✅ Saved to questions.json")
