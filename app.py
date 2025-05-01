import streamlit as st
import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle

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

# Ensure the 'static/uploads' directory exists
upload_folder = "static/uploads"
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Function to predict letter from image
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

# Streamlit Interface
st.title("Body Tracking and Letter Prediction")

# Upload multiple files
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    predictions = []
    
    # Save uploaded files to static/uploads
    for uploaded_file in uploaded_files:
        # Save each uploaded file to the correct path
        img_path = os.path.join(upload_folder, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Predict the letter from the saved image
        letter = predict_letter(img_path)
        predictions.append({'filename': uploaded_file.name, 'letter': letter})

    # Display predictions
    st.write("Predictions:")
    for pred in predictions:
        st.image(os.path.join(upload_folder, pred['filename']), caption=pred['filename'])
        st.write(f"Predicted Letter: {pred['letter']}")

    # Combine all predicted letters into a word
    word = ''.join([item['letter'] for item in predictions])
    st.write(f"Predicted Word: {word}")
