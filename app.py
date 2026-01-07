import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from PIL import Image

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Cat Emotion AI",
    page_icon="https://img.icons8.com/ios-filled/50/ffffff/cat.png",
    layout="centered"
)

# ------------------ LOAD MODELS (CACHED) ------------------
@st.cache_resource
def load_models():
    img_model = tf.keras.models.load_model("cat_image_model.keras")
    aud_model = tf.keras.models.load_model("cat_audio_model.keras")
    return img_model, aud_model

image_model, audio_model = load_models()

IMAGE_CLASSES = ['Happy','Sad','Angry','Surprised','Scared','Disgusted','Normal']
AUDIO_CLASSES = ['Happy','Angry','Paining','Resting','Warning','Fighting','Mating','Defense','HuntingMind','MotherCall']

# ------------------ STYLE ------------------
st.markdown("""
<style>
body {
    background: linear-gradient(180deg, #05010a, #14001f);
}
.stApp {
    background: linear-gradient(180deg, #05010a, #14001f);
    color: white;
}
.sidebar .sidebar-content {
    background: #0b0014;
}
.card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(14px);
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 0px 40px rgba(170,0,255,0.15);
    margin-bottom: 25px;
}
h1,h2,h3 {
    color: #e2c8ff;
}
.stButton>button {
    background: linear-gradient(135deg, #7b2cff, #b600ff);
    color: white;
    border-radius: 12px;
    padding: 12px 30px;
    font-size: 16px;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #b600ff, #7b2cff);
}
.result-box {
    background: linear-gradient(135deg, #1a002b, #2c0045);
    border-left: 6px solid #b600ff;
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
    box-shadow: 0px 0px 25px rgba(182,0,255,0.6);
}
.cat-icon {
    position: fixed;
    opacity: 0.07;
    z-index: 0;
}
.cat1 { top: 20px; left: 40px; width: 60px; }
.cat2 { bottom: 40px; right: 50px; width: 70px; }
.cat3 { top: 50%; right: 20px; width: 45px; }
</style>

<img src="https://img.icons8.com/ios-filled/100/ffffff/cat.png" class="cat-icon cat1">
<img src="https://img.icons8.com/ios/100/ffffff/cat-footprint.png" class="cat-icon cat2">
<img src="https://img.icons8.com/ios/100/ffffff/pet-commands.png" class="cat-icon cat3">
""", unsafe_allow_html=True)

# ------------------ PREPROCESS ------------------
def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((224,224))
    img = np.array(img)/255.0
    return np.expand_dims(img, axis=0)

def preprocess_audio(file):
    y, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return np.expand_dims(mfcc, axis=(0,1))

# ------------------ NAV ------------------
menu = st.sidebar.radio("Navigation", ["Home","Predict","About"])

# ------------------ HOME ------------------
if menu == "Home":
    st.title("Cat Emotion AI Platform")

    st.markdown("""
    <div class="card">
    This platform uses artificial intelligence to understand cat emotions from images and sound.
    It helps pet owners recognize stress, happiness, fear and other emotional states in their cats.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    This system was built to improve pet care by using computer vision and audio signal processing.
    By combining both modalities, the model can understand feline behavior more accurately.
    </div>
    """, unsafe_allow_html=True)

# ------------------ PREDICT ------------------
if menu == "Predict":
    st.title("Emotion Detection")

    st.markdown("<div class='card'>Upload a cat image, audio, or both.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        image_file = st.file_uploader("Upload Cat Image", type=["jpg","jpeg","png"])
        if image_file:
            st.image(image_file, use_column_width=True)

    with col2:
        audio_file = st.file_uploader("Upload Cat Audio", type=["wav","mp3"])

    if st.button("Analyze"):

        if not image_file and not audio_file:
            st.warning("Upload at least one input")
        else:
            with st.spinner("Processing"):
                if image_file:
                    img = preprocess_image(image_file)
                    img_pred = image_model.predict(img)[0]
                    img_emotion = IMAGE_CLASSES[np.argmax(img_pred)]

                if audio_file:
                    aud = preprocess_audio(audio_file)
                    aud_pred = audio_model.predict(aud)[0]
                    aud_emotion = AUDIO_CLASSES[np.argmax(aud_pred)]

            st.markdown("<div class='card'>", unsafe_allow_html=True)

            if image_file:
                st.markdown(f"""
                <div class="result-box">
                IMAGE EMOTION<br>
                {img_emotion}
                </div>
                """, unsafe_allow_html=True)

            if audio_file:
                st.markdown(f"""
                <div class="result-box">
                AUDIO EMOTION<br>
                {aud_emotion}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

# ------------------ ABOUT ------------------
if menu == "About":
    st.title("About This Project")

    st.markdown("""
    <div class="card">
    Cat Emotion AI is a deep learning based system that identifies emotional states in cats.
    It uses convolutional neural networks for images and audio feature extraction using MFCC for sound.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    The goal of this project is to help cat owners, veterinarians and animal researchers
    better understand feline behavior using artificial intelligence.
    </div>
    """, unsafe_allow_html=True)





