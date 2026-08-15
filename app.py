import os
import tempfile
import time
import cv2
import numpy as np
from PIL import Image
import streamlit as st

from config import DEFAULT_CONFIDENCE, DEFAULT_FRAME_WIDTH
from core.detector import Detector
from core.preprocessor import Preprocessor
from core.ocr_engine import OCREngine
from core.utils import FPSCounter, AlertManager, format_detection_log

# Streamlit Page Config
st.set_page_config(
    page_title="VisionGuard — Industrial Safety System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design Aesthetics
st.markdown("""
<style>
    /* Dark Theme Customizations */
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
    }
    
    /* Header Banner */
    .header-container {
        background: linear-gradient(135deg, #1E2640 0%, #0F172A 100%);
        padding: 24px 32px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38BDF8;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Badge Pills */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        background: #334155;
        color: #F8FAFC;
    }
    .badge-cyan { background: #0284C7; }
    .badge-emerald { background: #059669; }
    .badge-amber { background: #D97706; }
    .badge-purple { background: #7C3AED; }
    .badge-rose { background: #E11D48; }

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_ocr_engine():
    """Cached initialization of EasyOCR engine."""
    return OCREngine(gpu=False)


def apply_preprocessing_pipeline(
    frame: np.ndarray,
    noise_reduction: bool,
    clahe_enable: bool,
    grayscale: bool,
    contours: bool
) -> np.ndarray:
    """Run selected OpenCV preprocessing operations on frame."""
    processed = frame.copy()

    if noise_reduction:
        processed = Preprocessor.reduce_noise(processed)
    if clahe_enable:
        processed = Preprocessor.histogram_equalization(processed)
    if grayscale:
        processed = Preprocessor.to_grayscale(processed)
    if contours:
        processed = Preprocessor.extract_contours(processed)

    return processed


def main():
    # Header Banner
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🛡️ VisionGuard</div>
        <div class="header-subtitle">Real-Time Industrial Safety, Defect Detection & OCR Engine</div>
        <div style="margin-top: 14px;">
            <span class="badge badge-cyan">YOLOv8</span>
            <span class="badge badge-emerald">OpenCV</span>
            <span class="badge badge-purple">Instance Segmentation</span>
            <span class="badge badge-rose">Object Tracking</span>
            <span class="badge badge-amber">EasyOCR</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("⚙️ System Settings")

    # Detection & Segmentation Parameters
    st.sidebar.subheader("🎯 Model & CV Engines")
    use_segmentation = st.sidebar.toggle("Instance Segmentation Mode (YOLOv8-Seg)", value=False)
    enable_tracking = st.sidebar.toggle("Multi-Object Tracking (ByteTrack)", value=True)
    enable_ocr = st.sidebar.toggle("OCR Serial/Tag Reader (EasyOCR)", value=False)

    confidence_thresh = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_CONFIDENCE,
        step=0.05
    )

    # Initialize Detector
    detector = Detector(confidence=confidence_thresh, use_segmentation=use_segmentation)
    ocr_engine = get_ocr_engine() if enable_ocr else None

    # Input Mode Selection
    st.sidebar.subheader("📥 Input Source")
    input_source = st.sidebar.radio(
        "Select Input Type:",
        options=["📷 Image Upload", "🎥 Video File Upload"]
    )

    # OpenCV Preprocessing Pipeline Controls
    st.sidebar.subheader("🔬 OpenCV Preprocessing")
    enable_resize = st.sidebar.checkbox("Resize Frame (800px)", value=True)
    enable_noise = st.sidebar.checkbox("Noise Reduction (Gaussian Blur)", value=False)
    enable_clahe = st.sidebar.checkbox("Histogram Equalization (CLAHE)", value=False)
    enable_grayscale = st.sidebar.checkbox("Grayscale Conversion", value=False)
    enable_contours = st.sidebar.checkbox("Contour & Edge Extraction", value=False)

    st.sidebar.markdown("---")
    st.sidebar.caption("VisionGuard v2.0 • Advanced CV & OCR System")

    # ==================== MODE 1: IMAGE UPLOAD ====================
    if input_source == "📷 Image Upload":
        st.subheader("📷 Image Analysis & Inspection")
        uploaded_file = st.file_uploader(
            "Upload an industrial workshop / factory floor image",
            type=["jpg", "jpeg", "png", "webp"]
        )

        if uploaded_file is not None:
            # Read Image
            image = Image.open(uploaded_file).convert("RGB")
            frame = np.array(image)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if enable_resize:
                frame_bgr = Preprocessor.resize_frame(frame_bgr, width=DEFAULT_FRAME_WIDTH)

            start_time = time.perf_counter()

            # Apply OpenCV Preprocessing
            preprocessed_frame = apply_preprocessing_pipeline(
                frame_bgr,
                noise_reduction=enable_noise,
                clahe_enable=enable_clahe,
                grayscale=enable_grayscale,
                contours=enable_contours
            )

            # Run YOLO Detection / Segmentation / Tracking
            annotated_frame, detections = detector.detect(
                preprocessed_frame,
                enable_tracking=enable_tracking,
                draw_boxes=True
            )

            # Run EasyOCR if enabled
            if enable_ocr and ocr_engine is not None and detections:
                detections = ocr_engine.scan_frame_detections(preprocessed_frame, detections)
                # Re-render boxes with OCR text badges
                annotated_frame = preprocessed_frame.copy()
                for d in detections:
                    annotated_frame = Preprocessor.draw_bounding_box(
                        annotated_frame,
                        bbox=d["bbox"],
                        label=d["label"],
                        conf=d["confidence"],
                        track_id=d.get("track_id"),
                        ocr_text=d.get("ocr_text")
                    )

            proc_time_ms = round((time.perf_counter() - start_time) * 1000, 1)

            # Convert BGR back to RGB for display
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            prep_rgb = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2RGB)

            # Top Metrics Bar
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Objects Detected", len(detections))
            with col2:
                persons = sum(1 for d in detections if d['label'].lower() == 'person')
                st.metric("Workers Tracked", persons)
            with col3:
                alerts = AlertManager.generate_alerts(detections)
                st.metric("Active Safety Alerts", len(alerts))
            with col4:
                st.metric("Inference Latency", f"{proc_time_ms} ms")

            st.markdown("---")

            # Image Views (Side-by-Side)
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.caption("🔬 Preprocessed OpenCV Stream")
                st.image(prep_rgb, use_container_width=True)

            with img_col2:
                st.caption("🎯 VisionGuard YOLOv8 + Segmentation Output")
                st.image(annotated_rgb, use_container_width=True)

            # Safety Alerts Section
            st.subheader("⚠️ Safety & Violation Alerts")
            if alerts:
                for alert in alerts:
                    st.warning(alert)
            else:
                st.success("✅ No safety violations detected in current frame.")

            # Detections Log Table
            if detections:
                st.subheader("📋 Detection & OCR Log Breakdown")
                df_log = format_detection_log(detections)
                st.dataframe(df_log, use_container_width=True)

        else:
            st.info("👆 Please upload an image above to begin inspection.")

    # ==================== MODE 2: VIDEO UPLOAD ====================
    elif input_source == "🎥 Video File Upload":
        st.subheader("🎥 Video Stream Inspection & Object Tracking")
        uploaded_video = st.file_uploader(
            "Upload an industrial video stream (.mp4, .avi, .mov)",
            type=["mp4", "avi", "mov"]
        )

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            st.info(f"Video stream loaded successfully. Total frames: {total_frames}")

            col_btn1, col_btn2 = st.columns([1, 4])
            start_btn = col_btn1.button("▶️ Start Live Inspection", type="primary")

            if start_btn:
                st.markdown("---")
                
                m_col1, m_col2, m_col3 = st.columns(3)
                p_fps = m_col1.empty()
                p_count = m_col2.empty()
                p_alerts = m_col3.empty()

                video_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                fps_counter = FPSCounter()
                frame_idx = 0
                all_video_detections = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    if enable_resize:
                        frame = Preprocessor.resize_frame(frame, width=DEFAULT_FRAME_WIDTH)

                    prep_frame = apply_preprocessing_pipeline(
                        frame,
                        noise_reduction=enable_noise,
                        clahe_enable=enable_clahe,
                        grayscale=enable_grayscale,
                        contours=enable_contours
                    )

                    annotated, detections = detector.detect(
                        prep_frame,
                        enable_tracking=enable_tracking,
                        draw_boxes=True
                    )

                    if enable_ocr and ocr_engine is not None and frame_idx % 10 == 0:
                        detections = ocr_engine.scan_frame_detections(prep_frame, detections)

                    all_video_detections.extend(detections)

                    current_fps = fps_counter.update()

                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(annotated_rgb, use_container_width=True)

                    p_fps.metric("Processing FPS", f"{current_fps} FPS")
                    p_count.metric("Frame Objects", len(detections))
                    p_alerts.metric("Frame Alert Count", len(AlertManager.generate_alerts(detections)))

                    if total_frames > 0:
                        progress_bar.progress(min(frame_idx / total_frames, 1.0))

                cap.release()
                os.unlink(tfile.name)
                st.success("🎉 Video stream processing complete!")

                if all_video_detections:
                    st.subheader("📊 Full Video Inspection Log")
                    st.dataframe(format_detection_log(all_video_detections[:50]), use_container_width=True)

        else:
            st.info("👆 Please upload a video file to run automated stream inspection.")


if __name__ == "__main__":
    main()
