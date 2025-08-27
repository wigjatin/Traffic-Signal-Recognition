import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    return img / 255

def getClassName(classNo):
    classes = {
        0: 'Speed Limit 20 km/h', 1: 'Speed Limit 30 km/h', 2: 'Speed Limit 50 km/h',
        3: 'Speed Limit 60 km/h', 4: 'Speed Limit 70 km/h', 5: 'Speed Limit 80 km/h',
        6: 'End of Speed Limit 80 km/h', 7: 'Speed Limit 100 km/h', 8: 'Speed Limit 120 km/h',
        9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
        14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
        17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
        23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
        26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
        29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
        32: 'End of all speed and passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
        35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left',
        38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory',
        41: 'End of no passing', 42: 'End of no passing by vehicles over 3.5 metric tons'
    }
    return classes.get(classNo, "Unknown")

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return getClassName(classIndex), confidence, img

st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦", layout="centered")

st.markdown("<h1 style='text-align: center;'>🚦 Traffic Sign Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a traffic sign image and let the trained CNN predict it.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)

    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, img)
    label, confidence, processed_img = model_predict(temp_path)

    with col2:
        st.image(processed_img.reshape(32, 32), caption="Preprocessed (Grayscale)", use_column_width=True, clamp=True)

    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.success(f"**{label}**")
    st.progress(float(confidence))
    st.info(f"Confidence: {confidence*100:.2f}%")

st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>👨‍💻 Built by <b>Jatin Wig</b> | "
    "<a href='https://github.com/wigjatin/Traffic-Signal-Recognition' target='_blank'>"
    "GitHub Repository</a></p>",
    unsafe_allow_html=True
)
