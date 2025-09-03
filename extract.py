import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import numpy as np
import cv2
import pytesseract
import json
from pprint import pprint
import re
from image_processing import ImageProcessing, NormalizeText

config = "--oem 1 --psm 7 -l vie"

data = []
pdf_path = "./input/Module GPDC-2.pdf"
pages = convert_from_path(pdf_path)


for i, page in enumerate(pages):
    page = np.array(page)

    if i == 0:
        header_height = 300
        footer_height = 300
        left_width = 150
    else:
        header_height = 175
        footer_height = 250
        left_width = 150

    page = page[header_height:-footer_height, left_width:, :]

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
        if h > 28 and w > 60:
            roi = page[y : y + h, x : x + w]
            blocks.append((y, roi))

    # Sort the content following the y coordinates
    blocks = sorted(blocks, key=lambda b: b[0])

    # Preprocess the text
    for _, roi in blocks:
        roi = ImageProcessing.to_grayscale(roi)
        roi = ImageProcessing.rescale(roi)
        roi = ImageProcessing.denoise(roi)

    # Extract text
    for _, roi in blocks:
        roi = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
        text = pytesseract.image_to_string(roi, config=config)
        text = text.replace("\n", " ")
        text = text.replace("ŒC.", "C.")
        text = text.replace("Œ.", "C.")
        text = text.replace("Á.", "A.")
        text = text.replace("À.", "A.")
        text = text.replace("lR,", "B.")
        text = text.replace("€.", "C.")
        text = text.replace("Ù.", "D.")
        text = text.replace("€C.", "C.")
        text = text.replace("_ _D.", "D.")
        text = text.replace("B,", "B.")
        text = text.replace("AÀ.", "A.")
        text = text.replace("R,", "B.")
        text = text.replace("AA.", "A.")

        data.append(text)


with open("data.txt", "w") as file:
    for line in data:
        file.write(line + "\n")

questions = []
i = 0

while i < len(data):
    line = data[i].strip()

    if re.match(r"^(\d+[^\s]*[.,]|[a-z]\d+[^\s]*[.,])", line):
        question_lines = [line]
        i += 1

        # Gom các dòng tiếp theo cho đến khi gặp "A."
        while i < len(data) and not data[i].strip().startswith("A."):
            question_lines.append(data[i].strip())
            i += 1

        question_text = " ".join(question_lines)

        options = {}
        for opt in ["A", "B", "C", "D"]:
            if i < len(data):
                normalized = NormalizeText.normalize_option(data[i])
                if normalized.startswith(opt + "."):
                    options[opt] = normalized
                    i += 1
                else:
                    options[opt] = ""
            else:
                options[opt] = ""

        questions.append({"Question": question_text, **options})
    else:
        i += 1

with open("./output/questions.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print("✅ Saved to questions.json")
