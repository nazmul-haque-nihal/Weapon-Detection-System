# src/main.py
import argparse
import cv2
from src.detector import WeaponDetector

def main():
    parser = argparse.ArgumentParser(description="Weapon Detection System")
    parser.add_argument('--mode', choices=['image', 'video', 'webcam'], default='webcam',
                        help='Processing mode: image, video, or real-time webcam')
    parser.add_argument('--input', type=str, help='Input file path (for image/video)')
    parser.add_argument('--output', type=str, default='output.jpg', help='Output file path')
    args = parser.parse_args()

    detector = WeaponDetector()

    if args.mode == 'image':
        image = cv2.imread(args.input)
        if image is None:
            print(f"[ERROR] Could not load image: {args.input}")
            return
        detections = detector.detect(image)
        annotated = detector.draw_boxes(image, detections)
        cv2.imwrite(args.output, annotated)
        print(f"Saved annotated image to {args.output}")

    elif args.mode == 'video':
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {args.input}")
            return
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.detect(frame)
            annotated = detector.draw_boxes(frame, detections)
            out.write(annotated)
        cap.release()
        out.release()
        print(f"Saved annotated video to {args.output}")

    elif args.mode == 'webcam':
        print("[INFO] Starting real-time weapon detection. Press 'q' to quit.")
        cap = cv2.VideoCapture(0)  # Use default camera
        if not cap.isOpened():
            print("[ERROR] Could not access webcam")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.detect(frame)
            annotated = detector.draw_boxes(frame, detections)
            cv2.imshow('Weapon Detection (Press Q to Quit)', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()