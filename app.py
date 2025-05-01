import streamlit as st
import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle

# Set the page configuration for a more polished appearance
st.set_page_config(page_title="Semaphore Recognition", page_icon="🔠", layout="wide")

# Load model
model = load_model('model/landmark_model.h5')
with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

key_points = {
    'left_hand': mp_pose.PoseLandmark.LEFT_WRIST,
    'right_hand': mp_pose.PoseLandmark.RIGHT_WRIST,
    'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
    'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
    'nose': mp_pose.PoseLandmark.NOSE,
}

# Prediksi 1 gambar
def predict_letter(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "?"
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

# Mengatur layout halaman
st.title("Semaphore Flag Recognition 🔠")
st.markdown("""
    <style>
        .big-font {
            font-size:40px !important;
            color: #4CAF50;
            text-align: center;
        }
        .header {
            color: #fff;
            background-color: #4CAF50;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# Menampilkan instruksi
st.markdown('<p class="big-font">Upload gambar semaphore untuk diterjemahkan ke huruf!</p>', unsafe_allow_html=True)

# Upload gambar
uploaded_files = st.file_uploader("Pilih gambar semaphore", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_files:
    predictions = []
    for uploaded_file in uploaded_files:
        # Simpan gambar sementara
        img_path = os.path.join("static/uploads", uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Prediksi
        letter = predict_letter(img_path)
        predictions.append(letter)

    # Menampilkan hasil
    word = ''.join(predictions)
    st.subheader(f"Hasil: {word}")

    # Menampilkan gambar-gambar yang diupload
    st.image([file for file in uploaded_files], caption=[file.name for file in uploaded_files], width=150)
