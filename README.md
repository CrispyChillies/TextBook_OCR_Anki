# TextBook OCR to Anki Flashcards

This project provides a workflow to extract multiple-choice questions from textbook PDFs using OCR (Optical Character Recognition) and save them in a structured format suitable for generating flashcards in Anki.

## Features

- Converts textbook PDF pages to images.
- Crops and preprocesses images to isolate question blocks.
- Uses OCR (via Tesseract) to extract question text and answer options.
- Saves extracted questions and answers into an Excel file for easy import into Anki.

## Usage

1. Place your textbook PDF (e.g., `Module GPDC-2.pdf`) in the project directory.
2. Run the Jupyter notebook `ocr.ipynb` to process the PDF and extract questions.
3. The extracted questions and answers will be saved in an Excel file (e.g., `questions.xlsx`).
4. You can import this Excel file into Anki to generate flashcards for study.

## Requirements

- Python 3.x
- Jupyter Notebook
- `pdf2image`
- `opencv-python`
- `pytesseract`
- `matplotlib`
- `pandas`
- `openpyxl`

Install dependencies with:

```sh
pip install pdf2image opencv-python pytesseract matplotlib pandas openpyxl
```
