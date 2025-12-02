import streamlit as st
import cv2
import numpy as np
from src.detector import WeaponDetector
from PIL import Image
import tempfile
import time

# --- Page Configuration ---
st.set_page_config(page_title="Weapon Detection System", page_icon="🔫", layout="wide")

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Load the YOLO model and cache it."""
    detector = WeaponDetector(model_path='best.pt')
    return detector

detector = load_model()

# --- UI Sidebar ---
st.sidebar.title("Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
detection_mode = st.sidebar.selectbox("Choose Detection Mode", ["Image", "Video", "Webcam"])

# --- Main Application ---
st.title("End-to-End Object Detection Application")
st.write("This application uses a custom-trained YOLOv8 model to detect weapons in real-time.")

# --- Processing Logic ---
def process_frame(frame, conf_threshold):
    """
    Detects weapons in a single frame, draws bounding boxes, and returns the annotated image
    along with the list of detections.
    """
    detections = detector.detect(frame, conf=conf_threshold)
    annotated_image = detector.draw_boxes(frame.copy(), detections)
    return annotated_image, detections

# --- UI Views based on Mode ---
if detection_mode == "Image":
    st.header("Image-based Detection")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        # No need to convert colors for PIL-based image, detector handles it

        with st.spinner('Detecting weapons...'):
            annotated_image, detections = process_frame(frame, confidence_threshold)
        
        st.image(annotated_image, caption="Processed Image", use_column_width=True)
        st.metric("Objects Detected", len(detections))
        
        with st.expander("Detection Details"):
            st.write(detections)

elif detection_mode == "Video":
    st.header("Video-based Detection")
    uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        st_stats = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, detections = process_frame(frame_rgb, confidence_threshold)
            
            st_frame.image(annotated_frame, caption="Video in process", use_column_width=True)
            st_stats.metric("Objects Detected", len(detections))
        
        cap.release()
        tfile.close()

elif detection_mode == "Webcam":
    st.header("Real-time Webcam Detection")
    
    # Using streamlit-webrtc would be more robust, but for this project, a simple loop will suffice.
    run = st.checkbox('Run Webcam')
    FRAME_WINDOW = st.image([])
    stats_placeholder = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Please grant access and ensure it's not in use by another app.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to grab frame from webcam. Please restart the webcam.")
                    break
                
                # Convert frame from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_frame, detections = process_frame(frame_rgb, confidence_threshold)
                
                FRAME_WINDOW.image(annotated_frame)
                stats_placeholder.metric("Objects Detected", len(detections))
                
                # Add a small delay to prevent the app from being too resource-intensive
                time.sleep(0.01)
            
            cap.release()
            st.info('Webcam stopped.')