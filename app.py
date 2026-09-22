import os
import cv2
import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# 1. Load the trained ResNet50 model
MODEL_PATH = 'resnet50_feature_extraction.keras'
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# 2. Load Haar Cascade for face detection
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
if not os.path.exists(CASCADE_PATH):
    # Fallback to OpenCV internal path if local file is missing
    CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("Warning: Haar Cascade XML failed to load. Will evaluate full images instead of cropped faces.")

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def analyze_emotion(uploaded_image):
    if uploaded_image is None:
        return None

    # uploaded_image is provided as an RGB numpy array by Gradio
    gray = cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2GRAY)

    # Detect faces
    roi_rgb = uploaded_image
    if not face_cascade.empty():
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) > 0:
            # Crop to the first detected face
            x, y, w, h = faces[0]
            roi_rgb = uploaded_image[y:y+h, x:x+w]

    # Preprocess image for ResNet50
    roi_resized = cv2.resize(roi_rgb, (48, 48), interpolation=cv2.INTER_AREA)
    roi_batch = np.expand_dims(roi_resized, axis=0).astype(np.float32)
    roi_preprocessed = preprocess_input(roi_batch)

    # Predict emotion probabilities
    predictions = model.predict(roi_preprocessed, verbose=0)[0]
    return {class_names[i]: float(predictions[i]) for i in range(len(class_names))}

# 3. Build Gradio UI
interface = gr.Interface(
    fn=analyze_emotion,
    inputs=gr.Image(label="Upload Image", type="numpy"),
    outputs=gr.Label(num_top_classes=7, label="Emotion Predictions"),
    title="DeepFER: Facial Emotion Recognition",
    description="Upload an image to detect facial expressions using the trained ResNet50 architecture."
    # allow_flagging has been completely removed
)

if __name__ == '__main__':
    # share=True creates a public link
    # theme is now passed directly into launch() in Gradio 6.0+
    interface.launch(share=True, theme=gr.themes.Soft())