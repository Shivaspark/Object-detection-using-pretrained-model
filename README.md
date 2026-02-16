# Real-Time Object Detection with MobileNet SSD & Caffe (Project #06)

The sixth project in my **AI/ML Learning Path**. This project focuses on **efficient deep learning** by using a lightweight MobileNet-SSD model optimized for real-time performance on CPU-based systems.

## 📌 Overview
This project utilizes the OpenCV `dnn` (Deep Neural Network) module to load a pre-trained Caffe model. The **MobileNet-SSD** architecture is specifically designed for mobile and embedded vision applications, providing a great balance between speed and detection accuracy.

## 🛠️ Tech Stack
* **Language:** Python
* **Library:** OpenCV (DNN Module), NumPy
* **Model Architecture:** MobileNet + SSD (Single Shot MultiBox Detector)
* **Framework:** Caffe
* **Dataset:** PASCAL VOC (20 classes including person, car, dog, bottle, etc.)

## ⚙️ How It Works
1. **Model Loading:** The script loads the `.prototxt` file (defining the network structure) and the `.caffemodel` file (containing the trained weights).
2. **Blob Preprocessing:** Input frames are resized to 300x300 and normalized via mean subtraction to prepare them for the network.
3. **Inference:** The processed "blob" is passed through the SSD network to generate detections.
4. **Filtering:** The code iterates through detections, filtering out low-confidence results and drawing bounding boxes with class labels.

## 🚀 Quick Start
1. **Clone & Install:**
   ```bash
   pip install opencv-python numpy
2. **Required Files:** Ensure `deploy.prototxt` and `mobilenet_iter_73000.caffemodel` are in your project directory.
3. **Run the Script:**
   ```bash
   python main.py
*Progress: Mastered the use of optimized pre-trained models for real-time inference.*
