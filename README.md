# 🧠 High-Accuracy Face Recognition Pipeline with Model Benchmarking

A research-grade real-time computer vision system for benchmarking multiple face detection models under a fixed **ArcFace** recognition pipeline using **DeepFace**.

This repository contains the full experimental framework used to evaluate how **detector choice affects face recognition accuracy, confidence, and real-time reliability**.

---

## 🔍 What this project does

Most face recognition systems focus only on the embedding model (ArcFace, FaceNet, etc.).  
This project studies the **entire pipeline** by asking:

> **How does the choice of face detector impact identity recognition?**

All detectors are evaluated under the **same ArcFace embedding model**, allowing controlled and fair comparison.

---

## 🧬 Implemented Face Detection Backends

| Detector | File |
|--------|------|
| YOLOv8 | `yolov8_arcface.py` |
| RetinaFace | `retinaface_arcface.py` |
| MTCNN | `mtcnn_arcface.py` |
| MediaPipe | `mediapipe_arcface.py` |
| FastMTCNN | `fastmtcnn_arcface.py` |
| SSD | `ssd_arcface.py` |
| Dlib | `dlib_arcface.py` |
| OpenCV Haar | `opencv_arcface.py` |

Each script uses the **same ArcFace recognition logic**, changing only the detection backend.

---

## 🏗️ System Architecture

Webcam<br>
↓<br>
Face Detector (YOLOv8 / RetinaFace / MTCNN / MediaPipe / SSD / Dlib / OpenCV / FastMTCNN)<br>
↓<br>
Face ROI Extraction<br>
↓<br>
ArcFace Embedding (DeepFace)<br>
↓<br>
Similarity Search<br>
↓<br>
Confidence-calibrated Identity Output


---

## ⚙️ Tech Stack

- Python  
- DeepFace  
- ArcFace  
- OpenCV  
- YOLOv8  
- MTCNN  
- RetinaFace  
- MediaPipe  
- Dlib  

---

## 🧪 What this system evaluates

- Effect of detector quality on ArcFace embeddings  
- False positives and false negatives  
- Recognition confidence stability  
- Real-time inference performance  

This makes the project useful for:

- Biometric systems  
- Surveillance pipelines  
- Face authentication  
- Applied computer vision research  

---

## 🚀 How to run

### 1️⃣ Install dependencies

pip install deepface opencv-python ultralytics retina-face mediapipe mtcnn dlib
## 2️⃣ Create a local face database


**No images are included in this repository** for privacy and ethical reasons.

---

## 3️⃣ Run any detector

python yolov8_arcface.py

Press Q to quit.

## 🧠 Research insight
A strong recognition model (ArcFace) alone is not enough.
Detection quality directly controls how well embeddings represent identity.

By fixing ArcFace and swapping detectors, this system reveals how pipeline design impacts real-world biometric reliability.

👩‍💻 Author

Arpita Pani

AI Engineer — Computer Vision & Biometrics

GitHub: https://github.com/Jexa07

LinkedIn: https://www.linkedin.com/in/arpitapani07


