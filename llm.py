from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "google/mt5-small"

# load với use_fast=False để tránh lỗi tiktoken
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)


def normalize_ocr_text(text: str) -> str:
    """
    Sử dụng LLM (Flan-T5) để chỉnh sửa text OCR.
    Ví dụ: 'ŒC.' -> 'C.', 'B,' -> 'B.'
    """
    prompt = f"""
    Bạn là một bộ sửa lỗi OCR. 
    Quy tắc:
    - Nếu bắt đầu bằng 'ŒC.', '€C.' hoặc 'C,' -> đổi thành 'C.'
    - Nếu bắt đầu bằng 'Á.' -> đổi thành 'A.'
    - Nếu bắt đầu bằng 'B,' -> đổi thành 'B.'
    - Nếu bắt đầu bằng 'D..' -> đổi thành 'D.'
    - Giữ nguyên phần còn lại của câu.

    Chỉ in ra kết quả sửa, không thêm gì khác.

    Văn bản: {text}
    """

    result = pipe(prompt, max_length=50, clean_up_tokenization_spaces=True)
    return result[0]["generated_text"].strip()


# ---- Test thử ----
samples = [
    "ŒC. Khi đến tuổi dậy thì",
    "B, Sau gãy xương đã lành",
    "D.. Trong - ngoài",
    "Á. Phim video",
]

for s in samples:
    corrected = normalize_ocr_text(s)
    print(f"Original: {s}")
    print(f"Corrected: {corrected}")
    print("-" * 40)
