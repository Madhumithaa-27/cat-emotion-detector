import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from PIL import Image

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="PAWMOOD", layout="centered")

# ----------------------------
# Load models
# ----------------------------
@st.cache_resource
def load_models():
    img = tf.keras.models.load_model("cat_image_model.keras")
    aud = tf.keras.models.load_model("cat_audio_model.keras")
    return img, aud

img_model, aud_model = load_models()

# ----------------------------
# CSS
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');

body, .stApp {
    background: radial-gradient(circle at top, #1a0033, #000);
    color: white;
}

.title {
    font-family: 'Poppins', sans-serif;
    font-size: 64px;
    text-align: center;
    color: #b38bff;
    letter-spacing: 3px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #d1c4ff;
}

.upload-box {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    margin-top: 20px;
}

.pred-box {
    background: linear-gradient(135deg, #2b0040, #140020);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    font-size: 28px;
    margin-top: 30px;
    border: 1px solid #b38bff;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("assetspaw.png.png", width=90)

st.markdown('<div class="title">PAWMOOD</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Can’t wait to detect your cat’s emotion in voice</div>', unsafe_allow_html=True)

# ----------------------------
# Mode
# ----------------------------
mode = st.radio("Detection Mode", ["Image only", "Audio only", "Image + Audio"], horizontal=True)

st.markdown('<div class="upload-box">', unsafe_allow_html=True)

img_file = None
aud_file = None

if mode in ["Image only", "Image + Audio"]:
    img_file = st.file_uploader("Upload Cat Image", type=["jpg","png","jpeg"])

if mode in ["Audio only", "Image + Audio"]:
    aud_file = st.file_uploader("Upload Cat Voice", type=["wav","mp3"])

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Preprocess
# ----------------------------
def prep_image(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    return np.expand_dims(img, axis=0)

def prep_audio(aud):
    y, sr = librosa.load(aud, duration=3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return np.expand_dims(mfcc, axis=0)

labels = ["Happy", "Angry", "Sad", "Relaxed"]

# ----------------------------
# Predict
# ----------------------------
if st.button("Detect Mood"):
    preds = []

    if img_file:
        image = Image.open(img_file)
        st.image(image, width=250)
        img_input = prep_image(image)
        p = img_model.predict(img_input)[0]
        preds.append(p)

    if aud_file:
        aud_input = prep_audio(aud_file)
        p = aud_model.predict(aud_input)[0]
        preds.append(p)

    if preds:
        final = np.mean(preds, axis=0)
        mood = labels[np.argmax(final)]

        st.markdown(f'<div class="pred-box">{mood}</div>', unsafe_allow_html=True)

        st.image("assetscat_icon.png.png", width=80)
