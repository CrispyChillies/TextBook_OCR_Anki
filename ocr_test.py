import easyocr

reader = easyocr.Reader(["vi"])
results = reader.readtext("sample.png")

for bbox, text, prob in results:
    print(f"{text} (conf: {prob:.2f})")
