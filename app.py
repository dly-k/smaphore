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

# Custom Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #fafafa;
        font-family: 'Poppins', sans-serif;
    }
    .title-text {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        color: #3a0ca3;
        margin-bottom: 10px;
    }
    .subtitle-text {
        font-size: 18px;
        text-align: center;
        color: #555;
        margin-bottom: 30px;
    }
    .subheader-text {
        font-size: 26px;
        font-weight: 600;
        color: #5f0f40;
        margin-bottom: 20px;
    }
    .footer-text {
        font-size: 14px;
        color: #aaa;
        text-align: center;
        margin-top: 50px;
    }
    .pred-label {
        font-size: 20px;
        font-weight: 600;
        text-align: center;
        color: #3a0ca3;
        margin-bottom: 8px;
    }
    .file-name {
        font-size: 14px;
        font-weight: 500;
        text-align: center;
        color: #4a148c;
        margin-top: 4px;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Judul
st.markdown('<div class="title-text">Semaphore Translator 📡</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Upload gambar posisi semaphore, dan sistem akan menerjemahkannya jadi huruf!</div>', unsafe_allow_html=True)

# Uploader
uploaded_files = st.file_uploader("📂 Upload Gambar Semaphore", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.markdown('<div class="subheader-text">📖 Hasil Prediksi</div>', unsafe_allow_html=True)

    word = ""
    max_columns = 6
    num_files = len(uploaded_files)
    num_rows = (num_files // max_columns) + 1 if num_files % max_columns != 0 else num_files // max_columns

    # Prediksi huruf dulu untuk semua gambar
    predictions = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        letter = predict_letter(image)
        predictions.append((uploaded_file.name, image, letter))
        word += letter

    # Tampilkan hasil prediksi kata
    st.markdown(f'<div class="pred-label">🔤 <strong>Kata Hasil Prediksi:</strong> {word}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Tampilkan grid gambar dan hasilnya
    idx = 0
    for _ in range(num_rows):
        cols = st.columns(max_columns)
        for col in cols:
            if idx < num_files:
                filename, image, letter = predictions[idx]
                with col:
                    st.markdown(f'<div class="pred-label">{letter}</div>', unsafe_allow_html=True)
                    st.image(image, use_container_width=True, output_format="JPEG", clamp=True)
                    st.markdown(f'<div class="file-name">{filename}</div>', unsafe_allow_html=True)
                idx += 1
            else:
                break

# Footer
st.markdown('<div class="footer-text">Dibuat dengan ❤️ oleh Kelompok 18</div>', unsafe_allow_html=True)
