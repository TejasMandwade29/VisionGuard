# VisionGuard v2.0 — Real-Time Industrial Safety, Defect Detection & OCR System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection%20%26%20Segmentation-00FFFF.svg)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Preprocessing-green.svg)](https://opencv.org/)
[![ByteTrack](https://img.shields.io/badge/ByteTrack-Multi--Object%20Tracking-purple.svg)](https://github.com/ifzhang/ByteTrack)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-Text%20Recognition-orange.svg)](https://github.com/JaidedAI/EasyOCR)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)

VisionGuard is an end-to-end Computer Vision platform engineered for industrial workshops, assembly lines, and factory floors. It combines **YOLOv8 Object Detection & Instance Segmentation**, **Multi-Object Tracking (ByteTrack)**, **OpenCV Image Preprocessing**, and **EasyOCR Text Recognition** with a responsive **Streamlit Dashboard**.

---

## 🚀 Key Capabilities

1. **Object Detection & Instance Segmentation**:
   - Detects workers, safety gear, tools, and industrial machinery.
   - Extracts pixel-exact **polygon masks** (YOLOv8-Seg) for precise defect surface area estimation.
2. **Multi-Object Tracking (ByteTrack)**:
   - Assigns persistent **Track IDs** (e.g., Worker #1, Worker #2) across video frames.
   - Prevents duplicate alerts and tracks temporal movement on factory floors.
3. **OCR Tag & Serial Reader (EasyOCR)**:
   - Crops bounding box Regions of Interest (ROIs) with OpenCV.
   - Applies adaptive thresholding to extract text from equipment serial tags, warning signs, and worker badges.
4. **OpenCV Preprocessing Pipeline**:
   - **Gaussian Noise Reduction** filter.
   - **CLAHE (Histogram Equalization)** for dark or uneven factory lighting.
   - **Canny Contour Extraction** to highlight component boundaries and cracks.
5. **Interactive Streamlit Dashboard**:
   - Image & Video Upload modes.
   - Live FPS counters, latency metrics, and safety alert banners.
   - Interactive detection log dataframe with Track IDs and OCR strings.

---

## ⚡ Quick Start Guide

### 1. Clone & Navigate
`ash
git clone https://github.com/TejasMandwade29/VisionGuard.git
cd VisionGuard
`

### 2. Install Dependencies
`ash
pip install -r requirements.txt
`

### 3. Launch Dashboard
`ash
streamlit run app.py
`

Access the dashboard at http://localhost:8501.

---

## 📁 Project Structure

`	ext
VisionGuard/
├── app.py              # Streamlit dashboard UI
├── config.py           # Configuration & color schemes
├── requirements.txt    # Project dependencies
├── README.md           # Documentation
├── core/
│   ├── detector.py     # YOLOv8 detection & tracking engine
│   ├── ocr_engine.py   # EasyOCR text recognition
│   ├── preprocessor.py # OpenCV image preprocessing
│   └── utils.py        # Helpers, alerts & log formatter
├── models/             # YOLOv8 weights (.pt files)
└── data/
    └── sample/         # Sample test images & videos
`

---

## 📊 Feature Comparison Matrix

| Feature | Technology | Industrial Application |
|---------|------------|------------------------|
| **Object Detection** | YOLOv8n | Worker & PPE bounding box detection |
| **Instance Segmentation** | YOLOv8n-Seg | Pixel-accurate defect & machinery polygon masks |
| **Multi-Object Tracking** | ByteTrack | Unique ID assignment (Worker #1) across video streams |
| **OCR Text Extraction** | EasyOCR + OpenCV ROI | Reading serial tags & safety signs |
| **Contrast Enhancement** | OpenCV CLAHE | Equalizing brightness in low-light workshop environments |