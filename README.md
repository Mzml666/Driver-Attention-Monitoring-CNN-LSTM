# Real-Time Drowsiness and Distraction Detection using Computer Vision (OpenCV & MediaPipe)

## Overview
Driver drowsiness and distraction are major contributors to road accidents worldwide.  
This project implements a **real-time driver monitoring system** that detects **drowsiness and distraction** using **computer vision–based facial landmark analysis**.

The system processes live video input, extracts facial and eye-related geometric features, estimates head pose, and classifies the driver’s state in real time. The project is aligned with **Advanced Driver Assistance Systems (ADAS)** and **Driver Monitoring Systems (DMS)** used in modern vehicles.

---

## Objectives
- Detect driver drowsiness in real time using eye closure patterns  
- Identify distraction caused by head movement  
- Estimate driver attention using facial landmarks and head pose  
- Generate alerts to enhance driving safety  

---

## Key Features
- Real-time video processing using OpenCV  
- Face and eye landmark detection using MediaPipe Face Mesh  
- Eye Aspect Ratio (EAR)–based drowsiness detection  
- Head pose estimation using solvePnP (Yaw & Pitch analysis)  
- Low-latency inference suitable for real-time systems  
- Modular architecture extendable to deep learning models  

---

## Technologies Used
- **Programming Language:** Python  
- **Computer Vision:** OpenCV  
- **Facial Landmark Detection:** MediaPipe Face Mesh  
- **Mathematical Modeling:** EAR & 3D Head Pose Geometry  
- **Libraries:** NumPy, Math  

---

## System Architecture
The system follows the pipeline below:

1. Capture real-time video stream using a webcam  
2. Detect face and facial landmarks using MediaPipe  
3. Extract eye landmarks and compute Eye Aspect Ratio (EAR)  
4. Estimate head pose using 3D–2D landmark mapping (solvePnP)  
5. Classify driver state as **Alert / Drowsy / Distracted**  
6. Trigger visual alerts on the video stream  

---

## Dataset
- Real-time webcam input  
- No pre-recorded dataset required  
- Facial landmarks extracted dynamically per frame  

---

## Results & Performance
- **Real-Time Performance:** ~18–22 FPS  
- **Latency:** Low, suitable for live monitoring  
- **Accuracy:** High reliability under normal lighting conditions  
- Robust detection for frontal and near-frontal face poses  

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
- Rule-based thresholds may require tuning for different drivers  

---

## Future Enhancements
- CNN-based eye-state classification  
- LSTM-based temporal fatigue modeling  
- Attention heatmap visualization  
- Integration with vehicle systems (CAN / alerts)  
- Deployment on embedded platforms  
- Transformer-based temporal behavior modeling  

---

## Author
**Mohd Muzammil**  
B.Tech Computer Science Engineering  
VIT Chennai  
