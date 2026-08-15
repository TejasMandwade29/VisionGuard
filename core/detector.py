import os
import numpy as np
from ultralytics import YOLO
from config import DEFAULT_MODEL_NAME, SEGMENTATION_MODEL_NAME, DEFAULT_CONFIDENCE, DEFAULT_TRACKER, CLASS_COLORS
from core.preprocessor import Preprocessor


class Detector:
    """YOLOv8 Engine supporting Object Detection, Instance Segmentation, and Multi-Object Tracking."""

    def __init__(
        self,
        model_path: str = None,
        confidence: float = DEFAULT_CONFIDENCE,
        use_segmentation: bool = False
    ):
        self.confidence = confidence
        self.use_segmentation = use_segmentation

        if model_path is not None and os.path.exists(model_path):
            self.model_path = model_path
        elif use_segmentation:
            self.model_path = SEGMENTATION_MODEL_NAME
        else:
            self.model_path = DEFAULT_MODEL_NAME

        print(f"[VisionGuard] Loading YOLO Engine ({'Segmentation' if use_segmentation else 'Detection'}) from: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.names = self.model.names

    def detect(
        self,
        frame: np.ndarray,
        enable_tracking: bool = False,
        draw_boxes: bool = True
    ) -> tuple[np.ndarray, list]:
        """
        Run inference or multi-object tracking on frame.
        Returns:
            annotated_frame (np.ndarray): Frame with custom bounding boxes / segmentation masks.
            detections (list[dict]): Detection objects with bbox, mask, track_id, conf, label.
        """
        if frame is None:
            return None, []

        # Run tracking or standard prediction
        if enable_tracking:
            results = self.model.track(
                source=frame,
                conf=self.confidence,
                persist=True,
                tracker=DEFAULT_TRACKER,
                verbose=False
            )
        else:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                verbose=False
            )

        detections = []
        annotated_frame = frame.copy()

        if len(results) > 0:
            res = results[0]
            boxes = res.boxes
            masks = res.masks if hasattr(res, 'masks') and res.masks is not None else None

            if boxes is not None:
                for idx, box in enumerate(boxes):
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    label = self.names.get(cls_id, f"class_{cls_id}")

                    # Extract Track ID if tracking is active
                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id[0].cpu().numpy())

                    # Extract Polygon Mask if segmentation is active
                    polygon = None
                    if masks is not None and len(masks.xy) > idx:
                        polygon = masks.xy[idx]

                    det_entry = {
                        "bbox": xyxy,
                        "confidence": conf,
                        "class_id": cls_id,
                        "label": label,
                        "track_id": track_id,
                        "polygon": polygon
                    }
                    detections.append(det_entry)

                    if draw_boxes:
                        color = CLASS_COLORS.get(label.lower(), CLASS_COLORS["default"])

                        # Draw segmentation mask polygon first if available
                        if polygon is not None and len(polygon) > 0:
                            annotated_frame = Preprocessor.draw_segmentation_mask(
                                annotated_frame,
                                polygon=polygon,
                                color=color
                            )

                        # Draw bounding box & track label
                        annotated_frame = Preprocessor.draw_bounding_box(
                            annotated_frame,
                            bbox=xyxy,
                            label=label,
                            conf=conf,
                            color=color,
                            track_id=track_id
                        )

        return annotated_frame, detections
