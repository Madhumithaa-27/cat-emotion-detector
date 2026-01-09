import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from PIL import Image
import base64

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="PAWMOOD – Know Your Cat",
    page_icon="assetscat_icon.png.png",
    layout="centered"
)

# ------------------ LOAD IMAGES ------------------
def load_bg(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

paw_bg = load_bg("assetspaw.png.png")
cat_icon = load_bg("assetscat_icon.png.png")

# ------------------ CSS ------------------
st.markdown(f"""
<style>
.stApp {{
    background:
        linear-gradient(rgba(255,247,200,0.95), rgba(255,235,200,0.95)),
        url("data:image/png;base64,{paw_bg}");
    background-repeat: repeat;
    background-size: 150px;
    color:#2B1B2E;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #FFD1E8, #FFE9F3);
}}

h1, h2, h3, p, label {{
    color: #2B1B2E;
    font-family: 'Trebuchet MS', sans-serif;
}}

.logo {{
    display:flex;
    align-items:center;
    gap:15px;
    font-size:52px;
    font-weight:900;
    color:#6A0572;
    background: linear-gradient(135deg,#FFD86B,#FF9ACB);
    padding: 18px 40px;
    border-radius: 22px;
    box-shadow:0px 10px 25px rgba(0,0,0,0.25);
}}

.card {{
    background: rgba(255,255,255,0.97);
    padding: 35px;
    border-radius: 22px;
    box-shadow: 0px 15px 30px rgba(0,0,0,0.18);
    margin-bottom: 30px;
}}

.upload-box {{
    background: linear-gradient(135deg, #FFF7FB, #FFE4F1);
    border-radius: 20px;
    padding: 30px;
    border: 2px dashed #FF9ACB;
    text-align:center;
    font-weight:700;
    color:#7A1F57;
}}

.result-box {{
    background: linear-gradient(135deg, #FFD86B, #FF9ACB);
    padding: 32px;
    border-radius: 22px;
    font-size: 32px;
    font-weight: 900;
    text-align: center;
    color:#4B0035;
    animation: pop 0.35s ease-out;
    box-shadow:0px 12px 25px rgba(0,0,0,0.25);
}}

@keyframes pop {{
  0% {{transform: scale(0.7); opacity: 0;}}
  100% {{transform: scale(1); opacity: 1;}}
}}

.stButton>button {{
    background: linear-gradient(135deg,#FF7EB3,#FFD86B);
    color:#4B0035;
    border-radius: 14px;
    padding: 14px 30px;
    font-size: 16px;
    font-weight:800;
    border:none;
}}
</style>
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
    st.markdown(f"""
    <div class="logo">
        <img src="data:image/png;base64,{cat_icon}" width="60">
        PAWMOOD
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>Know your cat’s mood using AI</h3>
    Cats express emotions through face and voice. PAWMOOD uses deep learning to unders
