import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
from PIL import Image

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

# Streamlit UI
st.title("Semaphore Letter Prediction")
st.write("Upload beberapa gambar untuk diprediksi:")

uploaded_files = st.file_uploader("Pilih gambar", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    predictions = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        letter = predict_letter(image)
        predictions.append(letter)

    word = ''.join(predictions)
    st.subheader("Predicted Word:")
    st.write(word)
