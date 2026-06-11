#                        Real-Time Object Detection using YOLOv8 & Streamlit
A Computer Vision AI application that detects objects in real time from images or live webcam feeds — powered by YOLOv8 and wrapped in a clean Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit) ![Domain](https://img.shields.io/badge/Domain-AI%20%26%20ML-green)

---

## 📌 Overview

Traditional object detection systems are slow, resource-heavy, and not user-friendly. This project solves that by providing a fast, accurate, and accessible web app that:

- Detects **80 common everyday objects** out of the box (COCO dataset)
- Draws **bounding boxes** with labels and **confidence scores**
- Supports both **image upload** and **live webcam** modes
- Runs in the browser — **no installation needed for end users**

---

## 🚀 Features

- 📷 Two input modes: **Image Upload** (JPG/JPEG/PNG) and **Live Webcam**
- 🎚️ Adjustable **confidence threshold** slider to filter detections
- 🏷️ Real-time **bounding box annotations** with color-coded labels
- 📊 **Detection results table** showing object names and confidence scores
- 📋 Sidebar listing all **80 detectable object categories**
- ⚡ Fast inference using the **YOLOv8 Nano** model (lightweight & efficient)
- ☁️ Fully deployable as a web app on **Hugging Face Spaces**

---

## 🛠️ Tech Stack

| Component | Tool / Library | Purpose |
|---|---|---|
| Programming Language | Python 3.10+ | Core development |
| Object Detection Model | YOLOv8 (Ultralytics) | AI model for detection |
| Web Framework | Streamlit | Interactive web UI |
| Image Processing | OpenCV (cv2) | Capture & process images/video |
| Image Handling | Pillow (PIL) | Open and convert image formats |
| Numerical Computing | NumPy | Array and matrix operations |
| Deployment | Hugging Face Spaces | Free cloud deployment |

---

## ⚙️ How It Works

```
User Input (Image / Webcam)
        ↓
OpenCV reads & converts to RGB NumPy array
        ↓
YOLOv8 model runs inference (single forward pass)
        ↓
Model returns bounding boxes, class labels, confidence scores
        ↓
Annotated image + results table displayed on Streamlit UI
```

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.10 or above — download from [python.org](https://python.org)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/realtime-object-detection.git
cd realtime-object-detection
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the App
```bash
streamlit run app.py
```

### Step 4 — Open in Browser
The app opens automatically at `http://localhost:8501`. If it doesn't, paste the URL manually.

---

## 🗂️ Project Structure

```
realtime-object-detection/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🧠 Detectable Objects (80 COCO Classes)

| Category | Examples |
|---|---|
| People & Animals | person, cat, dog, bird, horse, cow, elephant, bear |
| Vehicles | car, bus, truck, motorcycle, bicycle, airplane, boat |
| Electronics | laptop, phone, TV, keyboard, mouse, remote |
| Kitchen Items | bottle, cup, fork, knife, spoon, bowl |
| Food | banana, apple, pizza, donut, sandwich, cake |
| Furniture | chair, couch, bed, dining table, toilet |
| Sports & Outdoors | sports ball, skateboard, surfboard, tennis racket |
| Other | book, clock, vase, scissors, backpack, suitcase |

---

## 📊 Expected Results

| Object | Predicted Label | Confidence |
|---|---|---|
| Water bottle | `bottle` | ~90–95% |
| Person | `person` | ~97%+ |
| Mobile phone | `cell phone` | High |
| Pen *(not in COCO)* | may misclassify | — |

---

---



## 📚 References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [Streamlit Docs](https://docs.streamlit.io)
- [OpenCV Docs](https://docs.opencv.org)
- [COCO Dataset](https://cocodataset.org)
- Redmon, J. et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR.
- Jocher, G. et al. (2023). *Ultralytics YOLOv8.* GitHub.

---

