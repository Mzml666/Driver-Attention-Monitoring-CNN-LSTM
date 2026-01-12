# Real-Time Drowsiness and Distraction Detection using CNN, LSTM, and OpenCV

## Overview
Driver drowsiness and distraction are major contributors to road accidents worldwide.  
This project implements a **real-time driver monitoring system** that detects **drowsiness and distraction** using computer vision and deep learning techniques.

The system processes live video input, extracts facial and eye-related features, models temporal behavior using deep learning, and classifies the driver’s state in real time. This project is aligned with **Advanced Driver Assistance Systems (ADAS)** and **Driver Monitoring Systems (DMS)** used in modern vehicles.

---

## Objectives
- Detect driver drowsiness in real time  
- Identify distraction caused by eye closure or head movement  
- Analyze temporal behavior using deep learning  
- Generate alerts to enhance driving safety  

---

## Key Features
- Real-time video processing using OpenCV  
- Face and eye detection  
- CNN-based spatial feature extraction  
- LSTM-based temporal sequence modeling  
- Low-latency inference suitable for real-time systems  
- Modular and scalable architecture  

---

## Technologies Used
- **Programming Language:** Python  
- **Computer Vision:** OpenCV  
- **Deep Learning Framework:** TensorFlow / Keras  
- **Models Used:**  
  - Convolutional Neural Networks (CNN)  
  - Long Short-Term Memory Networks (LSTM)  
- **Libraries:** NumPy, Matplotlib  

---

## System Architecture
The system follows the pipeline below:

1. Capture real-time video stream using a webcam  
2. Detect face and eyes using OpenCV  
3. Extract spatial features using CNN  
4. Model temporal eye and head movement patterns using LSTM  
5. Classify driver state as **Alert / Drowsy / Distracted**  
6. Trigger visual or audio alerts  

---

## Dataset
- Combination of **real-time webcam data** and **publicly available datasets**
- Dataset includes:
  - Eye open and closed states  
  - Head pose variations  
  - Distraction patterns  
- Labels used:
  - Alert  
  - Drowsy  
  - Distracted  

---

## Results & Performance
- **Accuracy:** ~90–92%  
- **Real-Time Performance:** ~18–22 FPS  
- **Latency:** Low, suitable for real-time deployment  
- Performs best under normal lighting conditions  

---
## Output
- Live video feed with driver state displayed  
- Alerts generated when drowsiness or distraction is detected  

---

## Applications
- Advanced Driver Assistance Systems (ADAS)  
- Driver Monitoring Systems (DMS)  
- Smart vehicle safety solutions  
- Fleet monitoring and management  
- Autonomous and semi-autonomous vehicles  

---

## Limitations
- Reduced performance in low-light conditions  
- Requires frontal or near-frontal face visibility  
- Not optimized for extreme head rotations  

---

## Future Enhancements
- Head pose estimation using 3D facial landmarks  
- Attention heatmap visualization  
- Integration with vehicle systems  
- Deployment on embedded platforms  
- Use of Transformer-based temporal models  

---

## Author
**Mohd Muzammil**  
B.Tech Computer Science Engineering  
VIT Chennai
