import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
from PIL import Image

# Load model
@st.cache_resource
def load_landmark_model():
    return load_model('model/landmark_model.h5')

@st.cache_resource
def load_label_encoder():
    with open('model/label_encoder.pkl', 'rb') as f:
        return pickle.load(f)

model = load_landmark_model()
label_encoder = load_label_encoder()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

key_points = {
    'left_hand': mp_pose.PoseLandmark.LEFT_WRIST,
    'right_hand': mp_pose.PoseLandmark.RIGHT_WRIST,
    'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
    'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
    'nose': mp_pose.PoseLandmark.NOSE,
}

def predict_letter(image):
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return "?"

    landmarks = results.pose_landmarks.landmark
    coords = []
    for idx in key_points.values():
        lm = landmarks[idx]
        coords.extend([lm.x, lm.y])

    left = landmarks[key_points['left_shoulder']]
    right = landmarks[key_points['right_shoulder']]
    coords.extend([(left.x + right.x) / 2, (left.y + right.y) / 2])

    X = np.array(coords).reshape(1, -1)
    prediction = model.predict(X)
    label_index = np.argmax(prediction)
    letter = label_encoder.inverse_transform([label_index])[0]
    return letter

# ======= Streamlit Custom Styling ======= #
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ece9e6, #ffffff);
        color: #333;
        font-family: 'Arial Rounded MT Bold', sans-serif;
    }
    .title-text {
        font-size:48px;
        font-weight:bold;
        text-align:center;
        color: #5a189a;
        margin-bottom: 20px;
    }
    .subtitle-text {
        font-size:22px;
        text-align:center;
        color: #6a4c93;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: #f3e9ff;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
        text-align:center;
    }
    </style>
""", unsafe_allow_html=True)

# ====== Judul & Deskripsi ====== #
st.markdown('<div class="title-text">🎨 Semaphore Letter Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Upload gambar semaphore kamu — biarkan AI menebak hurufnya 🎌</div>', unsafe_allow_html=True)

# File uploader
uploaded_files = st.file_uploader("📂 Pilih gambar semaphore", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("📖 Hasil Prediksi:")
    word = ""
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)
        letter = predict_letter(image)
        word += letter
        st.markdown(f'<div class="prediction-box">✨ <strong>Predicted Letter:</strong> {letter}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="prediction-box" style="background-color:#d0bdf4;"><strong>🔤 Predicted Word:</strong> {word}</div>', unsafe_allow_html=True)
