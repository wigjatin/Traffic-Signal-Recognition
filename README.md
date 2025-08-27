# Traffic Signal Recognition with Deep Learning

An intelligent computer vision system leveraging deep learning to classify and recognize traffic signals from images. Built with TensorFlow, OpenCV, and Streamlit for real-time interaction and testing.

---

## Problem Statement

Accurate traffic signal recognition is critical for autonomous vehicles and advanced driver-assistance systems (ADAS). Human perception can be error-prone, especially in fast-changing traffic environments. This model automates recognition using:

- Preprocessing and feature extraction with OpenCV
- Convolutional Neural Networks (CNNs) for classification
- Real-time deployment with Streamlit
- Cloud-ready design for reproducibility and scalability

---

## Key Features

- Grayscale and normalization preprocessing for noise-free inputs
- CNN-based classification trained on labeled traffic sign dataset
- Robust prediction pipeline using TensorFlow/Keras
- Lightweight deployment with opencv-python-headless for cloud environments
- Interactive Streamlit app for testing with uploaded images

---

## Model Architecture
```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(30,30,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

---

## Demo

You can access the live demo of the application by visiting the following link:  
[View Demo](https://traffic-signal-recognition-jatin-wig.streamlit.app/)

