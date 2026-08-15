import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "sample")

# Create required directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# YOLO Model Configurations
DEFAULT_MODEL_NAME = "yolov8n.pt"
SEGMENTATION_MODEL_NAME = "yolov8n-seg.pt"
DEFAULT_CONFIDENCE = 0.5

# Tracking Options
DEFAULT_TRACKER = "bytetrack.yaml"  # bytetrack.yaml or botsort.yaml

# OCR Configuration
OCR_LANGUAGES = ['en']
OCR_MIN_CONFIDENCE = 0.4


# Class names & Colors (BGR format for OpenCV drawing)
# Standard COCO person class ID is 0
COCO_PERSON_CLASS_ID = 0

CLASS_COLORS = {
    "person": (255, 191, 0),        # Deep Blue/Cyan
    "helmet": (0, 255, 0),          # Green
    "vest": (0, 215, 255),          # Gold/Yellow
    "goggles": (255, 0, 255),       # Magenta
    "no_helmet": (0, 0, 255),       # Red
    "no_vest": (0, 0, 255),         # Red
    "defect": (0, 165, 255),        # Orange
    "default": (200, 200, 200)      # Gray
}

# Image Preprocessing Settings
DEFAULT_FRAME_WIDTH = 800
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
GAUSSIAN_BLUR_KERNEL = (5, 5)
CANNY_THRESH1 = 50
CANNY_THRESH2 = 150
