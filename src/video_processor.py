import cv2
from src.detector import WeaponDetector

class VideoProcessor:
    def __init__(self, detector):
        self.detector = detector

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            detections = self.detector.detect(frame)
            annotated_frame = self.detector.draw_boxes(frame.copy(), detections)
            out.write(annotated_frame)
            
        cap.release()
        out.release()