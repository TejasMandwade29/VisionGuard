import cv2
import numpy as np
import easyocr
from config import OCR_LANGUAGES, OCR_MIN_CONFIDENCE


class OCREngine:
    """Optical Character Recognition (OCR) engine for industrial serial numbers and hazard tags."""

    def __init__(self, languages: list = OCR_LANGUAGES, gpu: bool = False):
        print(f"[VisionGuard] Initializing EasyOCR Reader for languages: {languages}")
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def extract_text(self, roi: np.ndarray) -> tuple[str, float]:
        """
        Extract text from a cropped region of interest (ROI).
        Returns:
            text (str): Extracted text string.
            confidence (float): Confidence score of OCR prediction.
        """
        if roi is None or roi.size == 0:
            return "", 0.0

        # OpenCV Preprocessing for OCR accuracy
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Contrast & Thresholding
        processed_roi = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        results = self.reader.readtext(processed_roi)
        
        # Fallback to grayscale if adaptive threshold yielded nothing
        if not results:
            results = self.reader.readtext(gray)

        extracted_words = []
        max_conf = 0.0

        for res in results:
            # res format: (bbox_corners, text, prob)
            if len(res) >= 3:
                text = res[1].strip()
                prob = float(res[2])
                if prob >= OCR_MIN_CONFIDENCE and len(text) > 1:
                    extracted_words.append(text)
                    max_conf = max(max_conf, prob)

        full_text = " ".join(extracted_words)
        return full_text, round(max_conf, 2)

    def scan_frame_detections(self, frame: np.ndarray, detections: list) -> list:
        """
        Perform OCR scan on all detected bounding box regions in frame.
        Appends 'ocr_text' field to detection objects.
        """
        if frame is None or not detections:
            return detections

        h, w = frame.shape[:2]

        for d in detections:
            bbox = d.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                # Clamp coordinates to frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if (x2 - x1) > 20 and (y2 - y1) > 15:
                    roi = frame[y1:y2, x1:x2]
                    text, conf = self.extract_text(roi)
                    if text:
                        d["ocr_text"] = text
                        d["ocr_conf"] = conf

        return detections
