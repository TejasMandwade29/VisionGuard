import cv2
import numpy as np
from config import (
    DEFAULT_FRAME_WIDTH,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    GAUSSIAN_BLUR_KERNEL,
    CANNY_THRESH1,
    CANNY_THRESH2,
    CLASS_COLORS
)


class Preprocessor:
    """OpenCV image preprocessing pipeline & advanced annotation renderer for VisionGuard."""

    @staticmethod
    def resize_frame(frame: np.ndarray, width: int = DEFAULT_FRAME_WIDTH) -> np.ndarray:
        """Resize frame maintaining aspect ratio."""
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            return frame
        aspect_ratio = h / w
        new_height = int(width * aspect_ratio)
        return cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def reduce_noise(frame: np.ndarray, kernel_size: tuple = GAUSSIAN_BLUR_KERNEL) -> np.ndarray:
        """Apply Gaussian Blur for noise reduction."""
        if frame is None:
            return None
        return cv2.GaussianBlur(frame, kernel_size, 0)

    @staticmethod
    def to_grayscale(frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to Grayscale (3-channel return for pipeline consistency)."""
        if frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def histogram_equalization(frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on Y channel of YUV."""
        if frame is None:
            return None
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def extract_contours(frame: np.ndarray) -> np.ndarray:
        """Extract edges using Canny and overlay contours onto frame."""
        if frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, CANNY_THRESH1, CANNY_THRESH2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output = frame.copy()
        cv2.drawContours(output, contours, -1, (0, 255, 255), 1)
        return output

    @staticmethod
    def draw_segmentation_mask(
        frame: np.ndarray,
        polygon: np.ndarray,
        color: tuple,
        alpha: float = 0.4
    ) -> np.ndarray:
        """Draw semi-transparent polygon mask for Instance Segmentation."""
        if frame is None or polygon is None or len(polygon) == 0:
            return frame

        overlay = frame.copy()
        pts = np.int32([polygon])
        cv2.fillPoly(overlay, pts, color)
        cv2.polylines(overlay, pts, True, color, 2)

        # Blend overlay with original frame
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    @staticmethod
    def draw_bounding_box(
        frame: np.ndarray,
        bbox: list,
        label: str,
        conf: float = None,
        color: tuple = None,
        track_id: int = None,
        ocr_text: str = None
    ) -> np.ndarray:
        """Draw styled bounding box with Object Track ID, Label, and OCR text."""
        if frame is None or len(bbox) < 4:
            return frame

        x1, y1, x2, y2 = map(int, bbox)

        if color is None:
            color = CLASS_COLORS.get(label.lower(), CLASS_COLORS["default"])

        # Main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Caption string
        caption_parts = []
        if track_id is not None:
            caption_parts.append(f"#{track_id}")
        caption_parts.append(label.capitalize())
        if conf is not None:
            caption_parts.append(f"{int(conf * 100)}%")

        caption = " ".join(caption_parts)

        # Draw main label pill
        (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lbl_y1 = max(y1 - text_h - 10, 0)
        lbl_y2 = y1

        cv2.rectangle(frame, (x1, lbl_y1), (x1 + text_w + 10, lbl_y2), color, -1)
        cv2.putText(
            frame,
            caption,
            (x1 + 5, y1 - 5 if y1 - 5 > text_h else y1 + text_h + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Draw secondary OCR text badge below bounding box if present
        if ocr_text:
            ocr_caption = f"📝 '{ocr_text}'"
            (ocr_w, ocr_h), _ = cv2.getTextSize(ocr_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y2), (x1 + ocr_w + 10, y2 + ocr_h + 8), (0, 0, 0), -1)
            cv2.putText(
                frame,
                ocr_caption,
                (x1 + 5, y2 + ocr_h + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

        return frame
