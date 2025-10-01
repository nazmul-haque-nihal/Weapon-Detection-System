# Advanced Multi-Class Weapon Detection System
<img width="646" height="579" alt="weapon2" src="https://github.com/user-attachments/assets/76cabd3c-d3d9-43af-bf7e-ce10ad2afc1e" />
<img width="644" height="579" alt="weapon5" src="https://github.com/user-attachments/assets/0bd9e998-946d-4387-97b1-db7986a44e28" />
<img width="646" height="575" alt="weapon3" src="https://github.com/user-attachments/assets/5c3b3730-ad4d-44ea-9ce7-0b6b60d888fe" />


## ✨ Features

- 🎯 **Multi-class Detection**: Automatic Rifle, Bazooka, Handgun, Knife, Grenade Launcher, Shotgun, SMG, Sniper, Sword
- 🚀 **Real-time Processing**: Optimized for live webcam feeds
- 🤖 **YOLOv8 Integration**: State-of-the-art object detection
- 📊 **Trained on 714 Images**: Custom weapon dataset with 9 classes
- 📹 **Multiple Input Modes**: Webcam, images, videos

## 📈 Model Performance

| Class | mAP50 | Instances | Status |
|-------|-------|-----------|--------|
| **Automatic Rifle** | 90.8% | 162 | ✅ Excellent |
| **Grenade Launcher** | 94.0% | 24 | ✅ Great |
| **Shotgun** | 99.0% | 21 | ✅ Perfect |
| **Bazooka** | 36.2% | 6 | ⚠️ Low data |
| **Knife** | 29.1% | 4 | ⚠️ Limited data |
| **Handgun** | 0.0% | 2 | ❌ Insufficient data |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/weapon-detection-system.git
cd weapon-detection-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install ultralytics opencv-python torch torchvision

# Install project packages
pip install -r requirements.txt
```

## 📁 Project Structure

```
weapon-detection-system/
├── data/                   # Datasets and test files
│   ├── weapon_detection/   # 9-class weapon dataset
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── val/
│   │       ├── images/
│   │       └── labels/
│   └── test/               # Sample test images
├── src/                    # Source code
│   ├── detector.py         # YOLOv8 weapon detector
│   └── main.py             # Main application entry point
├── runs/                   # Training outputs
│   └── detect/             # Model checkpoints and logs
└── utils/                  # Utility scripts
```

---

## ⚙️ Usage

### Real-time Webcam Detection
```bash
python -m src.main --mode webcam
```

### Image Detection
```bash
python -m src.main --mode image --input data/test/army.jpg --output result.jpg
```

### Video Processing
```bash
python -m src.main --mode video --input input.mp4 --output output.mp4
```

---

## 🤖 Model Architecture
- **Base Model**: YOLOv8n (Nano version for efficiency)
- **Classes**: 9 weapon types
- **Input Size**: 640x640
- **Output**: Bounding boxes with confidence scores
- **Training**: 100 epochs on custom dataset

---

## 📊 Training Details
```bash
yolo detect train \
  data=data/weapon9.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=8 \
  name=weapon9_final2
```

- **Total Images**: 714 (571 train + 143 val)
- **Training Time**: ~8.2 hours on CPU
- **Final Model**: `runs/detect/weapon9_final2/weights/best.pt`

---

## 🔧 Configuration

### Dataset Configuration (`data/weapon9.yaml`)
```yaml
train: ../data/weapon_detection/train/images
val: ../data/weapon_detection/val/images
nc: 9
names: ['Automatic Rifle', 'Bazooka', 'Grenade Launcher', 'Handgun', 'Knife', 'Shotgun', 'SMG', 'Sniper', 'Sword']
```

### Detector Settings (`src/detector.py`)
- **Confidence Threshold**: `0.3` (adjustable)
- **Classes**: All 9 weapon types
- **Output Format**: `xmin, ymin, xmax, ymax, confidence, class name`

---

## 🎯 Use Cases
- **Security Systems** – Real-time threat detection
- **Surveillance** – Automated weapon monitoring
- **Access Control** – Perimeter security
- **Law Enforcement** – Tactical operations support
- **Public Safety** – Event security monitoring

---

## 🚨 Limitations
- **CPU Training** – Model trained on CPU (slower performance)
- **Dataset Size** – Relatively small (714 images)
- **Class Imbalance** – Some classes underrepresented
- **Hardware** – Optimized for CPU inference (slower than GPU)

---

## 📈 Future Improvements
- Expand dataset with more samples
- GPU-based training for speed and accuracy
- Model quantization for edge deployment
- Multi-threaded real-time processing
- Integration with alert/notification systems
- Web-based monitoring dashboard

---

## 🏷️ Model Performance Metrics
- **Overall mAP50**: 58.2%
- **Overall mAP50-95**: 48.0%
- **Inference Speed**: ~87.7ms per image (CPU)
- **Model Size**: 6.3MB (compressed)

---

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
```bash
git checkout -b feature/your-feature-name
```
3. Make your changes
4. Commit your changes
```bash
git commit -m "Add your feature description"
```
5. Push to your branch
```bash
git push origin feature/your-feature-name
```
6. Open a Pull Request

---

## 📄 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments
- **Ultralytics YOLOv8** for the detection framework
- **Kaggle Dataset**: [snehilsanyal/weapon-detection-test](https://www.kaggle.com/datasets/snehilsanyal/weapon-detection-test)
- **Community** – Open-source research on weapon detection

---

## 🚀 Quick Start
```bash
# Activate environment
source venv/bin/activate

# Run real-time detection
python -m src.main --mode webcam

# Press 'q' to quit webcam feed
```

---

<p align="center">
  Built with ❤️ for public safety and security applications
</p>
