import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array#type:ignore

# Load the trained deepfake detection model
MODEL_PATH = "deepfake_detector.h5"  # Make sure this file exists in your project folder

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

# Function to predict image
def predict_image(image):
    try:
        # Resize image to match model input size (150x150)
        image = cv2.resize(image, (150, 150))  
        image = img_to_array(image) / 255.0  # Normalize to [0,1]
        image = np.expand_dims(image, axis=0)  # Expand dimensions for batch size

        # Make prediction
        prediction = model.predict(image)[0][0]

        return "Real" if prediction > 0.5 else "Fake"
    except Exception as e:
        return "Error"

# Streamlit UI
st.title("Deepfake Detection System")
st.write("Upload an image to check if it is real or fake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model_loaded:
        # Predict
        label = predict_image(image)
        st.write(f"### Prediction: {label}")
    else:
        st.error("Model is not loaded properly. Check the model file path.")
