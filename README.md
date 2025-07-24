# 📌 Object Tracking with Angle and Distance using Depth Anything V2

This is a **proof-of-concept project** that estimates the **distance** and **angle** of a detected human face in real-time using:

- [MediaPipe Face Detection](https://developers.google.com/mediapipe/solutions/vision/face_detection)
- [Depth Anything V2](https://github.com/isl-org/Depth-Anything/tree/main/depth_anything_v2)

It combines **depth estimation** and **facial detection** to infer the 3D position of a person from a monocular camera feed. The distance is **not directly measured**, but is inferred using a custom **exponential mapping** function based on prior observations.

---

## ⚙️ Features

- Real-time face detection using MediaPipe
- Distance estimation using a smoothed exponential function on raw depth values
- Horizontal and vertical angle estimation from face center
- Compatible with both IP cameras and USB webcams
- Frame saving to disk every 10 frames
- Logging of raw depth vs known distance for future curve fitting

---

## 🖥️ Hardware & Setup Requirements

- IP camera with snapshot URL or a USB webcam
- Python 3.8+ (run inside a Conda environment recommended)
- CUDA-capable GPU (for real-time performance)
- Internet access (if using IP cam)
- Depth Anything V2 model checkpoint

---

## 🚀 Performance Notes

- Tested on a machine with **RTX 4070 Ti Super**, achieving **20–25 FPS** at 1080p.
- The model can run on CPU, but frame rate will drop significantly (slow enough to hinder real-time tracking).
- GPU acceleration via **CUDA** is **strongly recommended** for acceptable performance.

---

## 🧪 Execution Disclaimer

The system works correctly, but **output images are not included** due to unavailability of the original camera hardware used during development.

All code remains executable and reproducible if you provide your own IP/webcam input and the required model checkpoint.

---

## 🛠️ Setup Instructions

### 1. Clone the repository & set up environment

```bash
git clone https://github.com/GSumanth109/object-tracking-with-angle-and-distance.git
cd object-tracking-with-angle-and-distance
conda create -n depthtrack python=3.9
conda activate depthtrack
pip install -r requirements.txt
