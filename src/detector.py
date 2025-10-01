# src/detector.py
from ultralytics import YOLO
import cv2

class WeaponDetector:
    def __init__(self, model_path='runs/detect/weapon9_final2/weights/best.pt', conf_threshold=0.3):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # All 9 weapon classes
        self.class_names = [
            'Automatic Rifle', 'Bazooka', 'Grenade Launcher', 'Handgun',
            'Knife', 'Shotgun', 'SMG', 'Sniper', 'Sword'
        ]

    def detect(self, image):
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                if cls_id >= len(self.class_names):
                    continue  # Safety check
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf.item())
                name = self.class_names[cls_id]
                detections.append({
                    'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                    'confidence': conf, 'name': name, 'class_id': cls_id
                })
        return detections

    def draw_boxes(self, image, detections):
        for det in detections:
            x1, y1, x2, y2 = det['xmin'], det['ymin'], det['xmax'], det['ymax']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{det['name']}: {det['confidence']:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return image