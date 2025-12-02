import gradio as gr
import cv2
import numpy as np
from src.detector import WeaponDetector
from PIL import Image
import os
import time

# --- Configuration ---
MODEL_PATH = 'runs/detect/weapon9_final2/weights/best.pt'
CONF_THRESHOLD = 0.3
THEME = 'gradio/monochrome'
TITLE = "Advanced Multi-Class Weapon Detection System"
DESCRIPTION = """
This application uses a **YOLOv8** model to detect **9 classes of weapons**:
*Automatic Rifle, Bazooka, Grenade Launcher, Handgun, Knife, Shotgun, SMG, Sniper, and Sword.*

You can test the model by uploading an image, a video file, or by starting the webcam feed. 
If a weapon is detected, the image will be automatically captured and displayed in the gallery below.
"""

# --- Detector Initialization ---
def load_model(model_path, conf_threshold):
    print("Loading model...")
    return WeaponDetector(model_path=model_path, conf_threshold=conf_threshold)

detector = load_model(MODEL_PATH, CONF_THRESHOLD)

# --- Processing Functions ---
def process_and_capture(image, captured_images):
    """
    Detects weapons, draws bounding boxes, captures the image if a weapon is found,
    and returns the annotated image, a summary, and the updated list of captured images.
    """
    if image is None:
        return None, "No image provided.", captured_images, captured_images

    # Convert PIL image to OpenCV format
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    detections = detector.detect(frame)
    annotated_image = detector.draw_boxes(frame.copy(), detections)
    
    # Create a summary of detections
    if detections:
        summary = "### Detected Weapons:\n"
        for det in detections:
            summary += f"- **{det['name']}** (Confidence: {det['confidence']:.2f})\n"
        
        # Capture the image if a weapon is detected
        if len(captured_images) < 20: # Limit the number of captures
            # Use annotated image for display in gallery
            captured_images.append(Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)))
            
    else:
        summary = "No weapons detected."
        
    # Convert BGR back to RGB for Gradio display
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return annotated_image_rgb, summary, captured_images, captured_images

# For video processing, the current implementation doesn't capture specific frames.
# This function will only return the processed video.
def process_video(video_path):
    """Processes a video file to detect weapons frame by frame."""
    if video_path is None:
        return None
        
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        annotated_frame = detector.draw_boxes(frame, detections)
        out.write(annotated_frame)
        
    cap.release()
    out.release()
    
    return output_path

# --- Gradio Interface ---
with gr.Blocks(theme=THEME) as demo:
    gr.Markdown(f"<h1 style='text-align: center;'>{TITLE}</h1>")
    gr.Markdown(DESCRIPTION)
    
    # State variable to hold the list of captured images
    captured_images_state = gr.State([])
    
    with gr.Tabs():
        # --- Image Detection Tab ---
        with gr.TabItem("Image Detection"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Image")
                    gr.Examples(
                        examples=[
                            os.path.join("data/test/test", "army.jpg"),
                            os.path.join("data/test/test", "terrorists.jpg"),
                            os.path.join("data/test/test", "weapons.jpg"),
                        ],
                        inputs=image_input
                    )
                with gr.Column():
                    image_output = gr.Image(label="Processed Image")
                    image_detection_summary = gr.Markdown()
            
            image_button = gr.Button("Detect Weapons in Image")

        # --- Video Detection Tab ---
        with gr.TabItem("Video Detection"):
            with gr.Row():
                video_input = gr.Video(label="Upload Video")
                video_output = gr.Video(label="Processed Video")
            
            video_button = gr.Button("Detect Weapons in Video")
            video_button.click(fn=process_video, inputs=video_input, outputs=video_output)

        # --- Webcam Detection Tab ---
        with gr.TabItem("Webcam Detection"):
            # State to control if webcam processing is active
            webcam_processing_active = gr.State(False) 

            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Webcam(streaming=True, label="Webcam Feed") # Changed from gr.Image
                with gr.Column():
                    webcam_output = gr.Image(label="Annotated Feed")
            
            with gr.Row():
                start_webcam_btn = gr.Button("Start Webcam Processing")
                stop_webcam_btn = gr.Button("Stop Webcam Processing")
                webcam_status = gr.Textbox(label="Webcam Status", value="Webcam processing stopped.", interactive=False)
    
    # --- Captured Detections Gallery ---
    with gr.Column():
        gr.Markdown("## Captured Detections")
        captured_gallery = gr.Gallery(label="Captured Images of Detected Weapons", columns=5)

    # --- Event Handlers ---
    image_button.click(
        fn=process_and_capture,
        inputs=[image_input, captured_images_state],
        outputs=[image_output, image_detection_summary, captured_images_state, captured_gallery]
    )

    # Webcam processing logic
    webcam_input.change(
        fn=process_and_capture,
        inputs=[webcam_input, captured_images_state],
        outputs=[webcam_output, gr.Markdown(), captured_images_state, captured_gallery],
        # The `fn` will only execute if webcam_processing_active is True
        # but the webcam_input still streams frames. We will manage processing internally.
        # This part requires careful management since `gr.Blocks` event listeners
        # don't directly have a 'condition' argument like `gr.Interface`.
    )

    # Start webcam processing button handler
    start_webcam_btn.click(
        lambda: True, 
        inputs=[], 
        outputs=webcam_processing_active
    ).success(
        lambda: "Webcam processing started. Keep the webcam tab open!", 
        inputs=[], 
        outputs=webcam_status
    )

    # Stop webcam processing button handler
    stop_webcam_btn.click(
        lambda: False, 
        inputs=[], 
        outputs=webcam_processing_active
    ).success(
        lambda: "Webcam processing stopped.", 
        inputs=[], 
        outputs=webcam_status
    )


if __name__ == "__main__":
    demo.launch()
