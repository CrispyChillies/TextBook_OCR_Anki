import cv2
import numpy as np
import re


class ImageProcessing:
    @staticmethod
    def to_grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def denoise(img, method="median"):
        if method == "median":
            return cv2.medianBlur(img, 3)
        elif method == "gaussian":
            return cv2.GaussianBlur(img, (3, 3), 0)
        else:
            raise ValueError("method must be 'median' or 'gaussian'")

    @staticmethod
    def rescale(img, target_char_height=40):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        heights = [
            cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[3] > 5
        ]

        if len(heights) > 0:
            avg_height = np.mean(heights)
            scale = float(target_char_height) / float(avg_height)
            return cv2.resize(
                img,
                None,
                fx=float(scale),
                fy=float(scale),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return img


class NormalizeText:
    @staticmethod
    def normalize_option(text):
        """
        Chuẩn hóa ký tự đầu đáp án:
        - Nhận A., Á., A.., A,  -> A.
        - Nhận B, B.. -> B.
        - Nhận ŒC., €C., C, -> C.
        - Nhận D.., Đ. -> D.
        """
        text = text.strip()

        # Regex tìm A/B/C/D ở đầu
        match = re.match(r"^[AÁAa][\.,]*", text)
        if match:
            return "A. " + text[len(match.group()) :].strip()

        match = re.match(r"^[Bb][\.,]*", text)
        if match:
            return "B. " + text[len(match.group()) :].strip()

        match = re.match(r"^[CcŒ€][\.,]*", text)
        if match:
            return "C. " + text[len(match.group()) :].strip()

        match = re.match(r"^[DdĐ][\.,]*", text)
        if match:
            return "D. " + text[len(match.group()) :].strip()

        return text  # fallback: giữ nguyên
