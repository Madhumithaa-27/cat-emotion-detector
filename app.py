import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from PIL import Image

st.set_page_config(page_title="Cat Emotion Detector 🐱", layout="centered")

# Load models
image_model = tf.keras.models.load_model("cat_image_model.keras")
audio_model = tf.keras.models.load_model("cat_audio_model.keras")

# Class labels (must match training)
IMAGE_CLASSES = ['Happy','Sad','Angry','Surprised','Scared','Disgusted','Normal']
AUDIO_CLASSES = ['Happy','Angry','Paining','Resting','Warning','Fighting','Mating','Defense','HuntingMind','MotherCall']

# ------------------- Preprocessing -------------------

def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_audio(file):
    y, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return np.expand_dims(mfcc, axis=(0,1))

# ------------------- UI -------------------

st.title("🐾 Cat Emotion Detection System")
st.markdown("Upload a **cat image** and **cat audio** to predict emotions.")

image_file = st.file_uploader("Upload Cat Image", type=["jpg","jpeg","png"])
audio_file = st.file_uploader("Upload Cat Audio", type=["wav","mp3"])

if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

if st.button("Detect Emotion"):
    if image_file and audio_file:
        with st.spinner("Analyzing..."):
            img = preprocess_image(image_file)
            aud = preprocess_audio(audio_file)

            img_pred = image_model.predict(img)[0]
            aud_pred = audio_model.predict(aud)[0]

            img_emotion = IMAGE_CLASSES[np.argmax(img_pred)]
            aud_emotion = AUDIO_CLASSES[np.argmax(aud_pred)]

        st.success("Prediction Complete 🎯")

        st.subheader("Image Result")
        st.write(f"Emotion: **{img_emotion}**")
        st.write(f"Confidence: **{round(np.max(img_pred)*100,2)}%**")

        st.subheader("Audio Result")
        st.write(f"Emotion: **{aud_emotion}**")
        st.write(f"Confidence: **{round(np.max(aud_pred)*100,2)}%**")

    else:
        st.warning("Please upload both image and audio")


