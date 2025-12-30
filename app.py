import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from PIL import Image

st.set_page_config(
    page_title="Cat Emotion Detector",
    page_icon="https://img.icons8.com/?size=100&id=9woxisYy4uwE&format=png&color=000000",
    layout="centered"
)

# ---------- Theme ----------
st.markdown("""
<style>
body {
    background-color: #F8F7BA;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #3A3A3A;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load models ----------
image_model = tf.keras.models.load_model("cat_image_model.keras")
audio_model = tf.keras.models.load_model("cat_audio_model.keras")

IMAGE_CLASSES = ['Happy','Sad','Angry','Surprised','Scared','Disgusted','Normal']
AUDIO_CLASSES = ['Happy','Angry','Paining','Resting','Warning','Fighting','Mating','Defense','HuntingMind','MotherCall']

# ---------- Preprocessing ----------
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

# ---------- Navigation ----------
menu = st.sidebar.radio("Navigation", ["About", "Predict"])

# ---------- ABOUT PAGE ----------
if menu == "About":
    st.image("https://img.icons8.com/?size=100&id=9woxisYy4uwE&format=png&color=000000", width=80)

    st.title("Cat Emotion Detection System")

    st.markdown("""
<div class="card">
This system uses deep learning to understand cat emotions from images and audio.

It analyzes facial expressions, posture, and sound features such as MFCC to classify emotional states like:
Happy, Angry, Sad, Resting, Warning, and more.

The goal is to help cat owners, veterinarians, and researchers understand cat behavior using AI.
</div>
""", unsafe_allow_html=True)

    st.image("https://images.unsplash.com/photo-1592194996308-7b43878e84a6", use_column_width=True)

# ---------- PREDICTION PAGE ----------
if menu == "Predict":

    st.title("Cat Emotion Prediction")

    st.markdown("<div class='card'>Upload a cat image or audio to analyze emotions.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        image_file = st.file_uploader("Upload Cat Image", type=["jpg","jpeg","png"])
        if image_file:
            st.image(image_file, use_column_width=True)

    with col2:
        audio_file = st.file_uploader("Upload Cat Audio", type=["wav","mp3"])

    if st.button("Detect Emotion"):

        if not image_file and not audio_file:
            st.warning("Upload at least one file")
        else:
            with st.spinner("Analyzing"):

                if image_file:
                    img = preprocess_image(image_file)
                    img_pred = image_model.predict(img)[0]
                    img_emotion = IMAGE_CLASSES[np.argmax(img_pred)]
                    img_conf = round(np.max(img_pred) * 100, 2)

                if audio_file:
                    aud = preprocess_audio(audio_file)
                    aud_pred = audio_model.predict(aud)[0]
                    aud_emotion = AUDIO_CLASSES[np.argmax(aud_pred)]
                    aud_conf = round(np.max(aud_pred) * 100, 2)

            st.markdown("<div class='card'>", unsafe_allow_html=True)

            if image_file:
                st.subheader("Image Result")
                st.write("Emotion:", img_emotion)
                st.write("Confidence:", img_conf, "%")

            if audio_file:
                st.subheader("Audio Result")
                st.write("Emotion:", aud_emotion)
                st.write("Confidence:", aud_conf, "%")

            st.markdown("</div>", unsafe_allow_html=True)

    st.image("https://images.unsplash.com/photo-1601758064131-59a3bca1f06b", use_column_width=True)



