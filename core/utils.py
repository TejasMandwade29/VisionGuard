import time
from collections import Counter


class FPSCounter:
    """Calculates smoothed frames per second (FPS)."""

    def __init__(self, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.frame_times = []
        self.last_time = time.perf_counter()

    def update(self) -> float:
        """Call on every frame update. Returns current FPS."""
        now = time.perf_counter()
        delta = now - self.last_time
        self.last_time = now

        if delta > 0:
            self.frame_times.append(delta)
            if len(self.frame_times) > self.buffer_size:
                self.frame_times.pop(0)

        if not self.frame_times:
            return 0.0

        avg_delta = sum(self.frame_times) / len(self.frame_times)
        return round(1.0 / avg_delta, 1)


class AlertManager:
    """Manages violation alerts and defect summary logs."""

    @staticmethod
    def generate_alerts(detections: list) -> list:
        """Analyze detections and return list of actionable alerts."""
        alerts = []
        counts = Counter([d["label"].lower() for d in detections])

        person_count = counts.get("person", 0)
        helmet_count = counts.get("helmet", 0)
        vest_count = counts.get("vest", 0)

        # General PPE heuristics if using PPE model
        if helmet_count > 0 or vest_count > 0:
            if person_count > helmet_count:
                missing = person_count - helmet_count
                alerts.append(f"⚠️ {missing} worker(s) missing Safety Helmet")
            if person_count > vest_count:
                missing = person_count - vest_count
                alerts.append(f"⚠️ {missing} worker(s) missing High-Vis Vest")

        # Explicit missing class detection
        if counts.get("no_helmet", 0) > 0:
            alerts.append(f"🚨 ALERT: {counts['no_helmet']} Safety Helmet violation(s) detected!")
        if counts.get("no_vest", 0) > 0:
            alerts.append(f"🚨 ALERT: {counts['no_vest']} High-Vis Vest violation(s) detected!")

        # COCO fallback notification if only detecting person
        if person_count > 0 and helmet_count == 0 and vest_count == 0 and "no_helmet" not in counts:
            alerts.append(f"ℹ️ {person_count} worker(s) detected in factory zone (General Monitoring Mode)")

        return alerts


def format_detection_log(detections: list) -> list:
    """Format raw detections into clean records for Streamlit table display."""
    records = []
    for idx, d in enumerate(detections, start=1):
        bbox_str = f"[{int(d['bbox'][0])}, {int(d['bbox'][1])}, {int(d['bbox'][2])}, {int(d['bbox'][3])}]"
        rec = {
            "#": idx,
            "Track ID": f"#{d['track_id']}" if d.get("track_id") is not None else "N/A",
            "Object Class": d["label"].capitalize(),
            "Confidence": f"{d['confidence'] * 100:.1f}%",
            "Bounding Box (x1,y1,x2,y2)": bbox_str,
            "OCR Extracted Text": d.get("ocr_text", "—")
        }
        records.append(rec)
    return records

