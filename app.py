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

# ------------------ DARK THEME ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0B0B14, #14142B);
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #0F0F1E;
}

.card {
    background: linear-gradient(145deg, #1C1C3A, #101022);
    padding: 28px;
    border-radius: 18px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.6);
    margin-bottom: 25px;
    border: 1px solid rgba(255,255,255,0.05);
}

.square {
    background: linear-gradient(135deg, #7F5AF0, #2CB67D);
    width: 160px;
    height: 160px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-weight: 700;
    color: white;
    text-align: center;
    margin: 10px auto;
}

.result-box {
    background: linear-gradient(135deg, #2B2B5E, #1B1B38);
    padding: 30px;
    border-radius: 18px;
    border-left: 8px solid #7F5AF0;
    font-size: 26px;
    font-weight: 800;
    text-align: center;
    color: #FFFFFF;
    animation: pop 0.4s ease-out;
    margin-top: 15px;
}

@keyframes pop {
  0% {transform: scale(0.7); opacity: 0;}
  100% {transform: scale(1); opacity: 1;}
}

.stButton>button {
    background: linear-gradient(135deg, #7F5AF0, #2CB67D);
    color: white;
    border-radius: 12px;
    padding: 14px 28px;
    font-size: 16px;
    font-weight: 700;
    border: none;
}

.icon-btn {
    display: flex;
    align-items: center;
    gap: 10px;
}

.cat-icon {
    width: 28px;
}

.cat-bg {
    position: fixed;
    opacity: 0.06;
    z-index: 0;
}

.cat1 { top: 40px; left: 30px; width: 70px; }
.cat2 { bottom: 50px; right: 40px; width: 80px; }
.cat3 { top: 50%; right: 20px; width: 50px; }
</style>

<img src="https://img.icons8.com/ios-filled/100/ffffff/cat.png" class="cat-bg cat1">
<img src="https://img.icons8.com/ios/100/ffffff/cat-footprint.png" class="cat-bg cat2">
<img src="https://img.icons8.com/ios/100/ffffff/pet-commands.png" class="cat-bg cat3">
""", unsafe_allow_html=True)

# ------------------ MODELS ------------------
image_model = tf.keras.models.load_model("cat_image_model.keras")
audio_model = tf.keras.models.load_model("cat_audio_model.keras")

IMAGE_CLASSES = ['Happy','Sad','Angry','Surprised','Scared','Disgusted','Normal']
AUDIO_CLASSES = ['Happy','Angry','Paining','Resting','Warning','Fighting','Mating','Defense','HuntingMind','MotherCall']

# ------------------ PREPROCESS ------------------
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

# ------------------ NAV ------------------
menu = st.sidebar.radio("Navigation", ["Home", "Predict", "About"])

# ------------------ HOME ------------------
if menu == "Home":
    st.title("Cat Emotion Detector")

    st.markdown("""
    <div class="card">
    This AI system understands how a cat feels by analyzing images and sounds.
    Cats cannot speak, but their face and voice reveal emotions.
    This platform helps cat owners recognize happiness, fear, pain, or stress early.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='square'>Pet emotions are valid</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='square'>AI for animal care</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='square'>Understand your cat</div>", unsafe_allow_html=True)

# ------------------ PREDICT ------------------
if menu == "Predict":

    st.title("Detect Emotion")

    st.markdown("<div class='card'>Upload a cat image or audio to analyze emotions.</div>", unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Cat Image", type=["jpg","jpeg","png"])
    audio_file = st.file_uploader("Upload Cat Audio", type=["wav","mp3"])

    if st.button("Detect Emotion"):

        if not image_file and not audio_file:
            st.warning("Upload at least one file")
        else:
            with st.spinner("Analyzing"):

                img_emotion = None
                aud_emotion = None

                if image_file:
                    img = preprocess_image(image_file)
                    img_pred = image_model.predict(img)[0]
                    img_emotion = IMAGE_CLASSES[np.argmax(img_pred)]

                if audio_file:
                    aud = preprocess_audio(audio_file)
                    aud_pred = audio_model.predict(aud)[0]
                    aud_emotion = AUDIO_CLASSES[np.argmax(aud_pred)]

            if img_emotion:
                st.markdown(f"<div class='result-box'>Image Emotion: {img_emotion}</div>", unsafe_allow_html=True)

            if aud_emotion:
                st.markdown(f"<div class='result-box'>Audio Emotion: {aud_emotion}</div>", unsafe_allow_html=True)

# ------------------ ABOUT ------------------
if menu == "About":
    st.title("About This Project")

    st.markdown("""
    <div class="card">
    This Cat Emotion Detection system was developed by Madhumithaa D K.
    It uses deep learning models trained on cat image and audio datasets
    to recognize emotional states such as happiness, fear, pain, aggression, and calmness.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    Technologies used include TensorFlow for model training,
    Librosa for audio processing,
    and Streamlit for the interactive web application.
    The goal is to support cat owners and veterinarians
    by providing early insight into a cat’s emotional health.
    </div>
    """, unsafe_allow_html=True)

