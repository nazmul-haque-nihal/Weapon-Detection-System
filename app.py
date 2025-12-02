import gradio as gr
import cv2
import numpy as np
from src.detector import WeaponDetector
from PIL import Image
import os
# No time.sleep needed as per user request for faster updates

# --- Configuration ---
MODEL_PATH = 'runs/detect/weapon9_final2/weights/best.pt'
CONF_THRESHOLD = 0.3
THEME = 'gradio/monochrome'
TITLE = "Advanced Multi-Class Weapon Detection System"
DESCRIPTION = """
This application uses a **YOLOv8** model to detect **9 classes of weapons**:
*Automatic Rifle, Bazooka, Grenade Launcher, Handgun, Knife, Shotgun, SMG, Sniper, and Sword.*

The live webcam will show detections in real-time.
"""

# --- Detector Initialization ---
detector = WeaponDetector(model_path=MODEL_PATH, conf_threshold=CONF_THRESHOLD)

# --- Processing Functions ---
def process_frame(image):
    """
    Processes a single frame (from image upload or webcam).
    """
    if image is None:
        return None, "Awaiting input..."

    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    detections = detector.detect(frame)
    annotated_image = detector.draw_boxes(frame.copy(), detections)
    
    summary = "### Detected Weapons:\n"
    if detections:
        for det in detections:
            summary += f"- **{det['name']}** (Confidence: {det['confidence']:.2f})\n"
    else:
        summary = "No weapons detected."
        
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Return only the annotated image and summary, no gallery updates
    return annotated_image_rgb, summary

def process_video(video_path):
    """Processes a video file to detect weapons frame by frame."""
    if video_path is None: return None
    cap = cv2.VideoCapture(video_path)
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
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
    
    # Removed captured_images_state and all gallery components
    # captured_images_state = gr.State([])

    with gr.Tabs():
        with gr.TabItem("Image Detection"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Image")
                    gr.Examples(examples=[os.path.join("data/test/test", "army.jpg"), os.path.join("data/test/test", "terrorists.jpg")], inputs=image_input)
                with gr.Column():
                    image_output = gr.Image(label="Processed Image")
                    image_detection_summary = gr.Markdown()
            image_button = gr.Button("Detect Weapons in Image")

        with gr.TabItem("Video Detection"):
            with gr.Row():
                video_input = gr.Video(label="Upload Video")
                video_output = gr.Video(label="Processed Video")
            video_button = gr.Button("Detect Weapons in Video")

        with gr.TabItem("Webcam Detection"):
            with gr.Row():
                webcam_input = gr.Image(source="webcam", streaming=True, label="Webcam Feed")
                webcam_output = gr.Image(label="Annotated Feed")
            webcam_summary = gr.Markdown("Awaiting input...")

    # Removed captured_gallery and associated markdown
    # with gr.Column():
    # gr.Markdown("## Captured Detections")
    # captured_gallery = gr.Gallery(label="Captured Images of Detected Weapons", columns=5, height="auto")

    # Event Handlers - simplified outputs
    image_button.click(
        fn=process_frame, 
        inputs=[image_input], 
        outputs=[image_output, image_detection_summary]
    )
    
    video_button.click(fn=process_video, inputs=video_input, outputs=video_output)
    
    webcam_input.change(
        fn=process_frame,
        inputs=[webcam_input],
        outputs=[webcam_output, webcam_summary]
    )

if __name__ == "__main__":
    demo.launch()
