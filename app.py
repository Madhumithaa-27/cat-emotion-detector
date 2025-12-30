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

# ------------------- THEME -------------------
st.markdown("""
<style>

.stApp {
    background-color: #F8F7BA;
}

[data-testid="stSidebar"] {
    background-color: #EFECA3;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

[data-testid="stFileUploader"] {
    background-color: #FFFBD9;
    border-radius: 12px;
    padding: 12px;
}

.stButton>button {
    background: linear-gradient(135deg, #FFB703, #FB8500);
    color: black;
    border-radius: 12px;
    padding: 12px 26px;
    font-size: 16px;
    font-weight: 700;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #FB8500, #FFB703);
}

.result-box {
    background: linear-gradient(135deg, #E0F7FA, #B2EBF2);
    padding: 22px;
    border-radius: 16px;
    border-left: 8px solid #00ACC1;
    margin-top: 15px;
    font-size: 18px;
    font-weight: 700;
    color: #004D40;
}

</style>
""", unsafe_allow_html=True)

# ------------------- LOAD MODELS -------------------
image_model = tf.keras.models.load_model("cat_image_model.keras")
audio_model = tf.keras.models.load_model("cat_audio_model.keras")

IMAGE_CLASSES = ['Happy','Sad','Angry','Surprised','Scared','Disgusted','Normal']
AUDIO_CLASSES = ['Happy','Angry','Paining','Resting','Warning','Fighting','Mating','Defense','HuntingMind','MotherCall']

# ------------------- PREPROCESS -------------------
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

# ------------------- NAVIGATION -------------------
menu = st.sidebar.radio("Navigation", ["About", "Predict"])

# ------------------- ABOUT -------------------
if menu == "About":
    st.title("Cat Emotion Detection System")

    st.markdown("""
<div class="card">
This system uses deep learning to understand cat emotions from images and audio.
It analyzes facial expressions, posture, and sound features such as MFCC to classify emotional states.
The goal is to help cat owners, veterinarians, and researchers understand cat behavior using AI.
</div>
""", unsafe_allow_html=True)

    st.image("https://images.unsplash.com/photo-1592194996308-7b43878e84a6", use_column_width=True)

# ------------------- PREDICT -------------------
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

            st.balloons()

            st.markdown("<div class='card'>", unsafe_allow_html=True)

            if image_file:
                st.markdown(f"""
<div class="result-box">
IMAGE RESULT<br>
Emotion: {img_emotion}<br>
Confidence: {img_conf} %
</div>
""", unsafe_allow_html=True)

            if audio_file:
                st.markdown(f"""
<div class="result-box">
AUDIO RESULT<br>
Emotion: {aud_emotion}<br>
Confidence: {aud_conf} %
</div>
""", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    st.image("https://images.unsplash.com/photo-1601758064131-59a3bca1f06b", use_column_width=True)
True)



