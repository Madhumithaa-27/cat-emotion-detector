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
    background-color: #FFF7C2;
    background-image: url("data:image/png;base64,{paw_bg}");
    background-repeat: repeat;
    background-size: 180px;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #FFB7D5, #FFE9F2);
}}

h1, h2, h3, p, label {{
    color: #2B1B2E;
    font-family: 'Trebuchet MS', sans-serif;
}}

.logo {{
    display:flex;
    align-items:center;
    gap:15px;
    font-size:48px;
    font-weight:900;
    color:#6A0572;
}}

.card {{
    background: white;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}}

.upload-box {{
    background: linear-gradient(135deg, #FFF6FB, #FFE4F1);
    border-radius: 18px;
    padding: 25px;
    border: 2px dashed #FF9ACB;
    text-align:center;
    font-weight:700;
    color:#7A1F57;
}}

.result-box {{
    background: linear-gradient(135deg, #FFB6D9, #FFD9EC);
    padding: 30px;
    border-radius: 20px;
    font-size: 30px;
    font-weight: 900;
    text-align: center;
    color: #4B0035;
    box-shadow: 0px 10px 20px rgba(0,0,0,0.25);
}}

.stButton>button {{
    background: linear-gradient(135deg, #FF7EB3, #FFB347);
    color: #2B1B2E;
    border-radius: 14px;
    padding: 14px 28px;
    font-size: 16px;
    font-weight: 800;
    border: none;
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
        <img src="data:image/png;base64,{cat_icon}" width="70">
        PAWMOOD
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>Know your cat’s mood using AI</h3>
    Cats express emotions through face and voice. PAWMOOD uses deep learning to understand what your cat is feeling — happiness, fear, stress or pain.
    </div>
    """, unsafe_allow_html=True)

# ------------------ PREDICT ------------------
if menu == "Predict":

    st.title("Emotion Detection")

    st.markdown("<div class='upload-box'>Upload a cat image or audio — PAWMOOD is excited to understand your cat</div>", unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Cat Image", type=["jpg","jpeg","png"])
    audio_file = st.file_uploader("Upload Cat Audio", type=["wav","mp3"])

    if st.button("Detect Emotion"):

        if not image_file and not audio_file:
            st.warning("Please upload at least one file")
        else:
            img_emotion = None
            aud_emotion = None

            if image_file:
                img = preprocess_image(image_file)
                pred = image_model.predict(img)[0]
                img_emotion = IMAGE_CLASSES[np.argmax(pred)]

            if audio_file:
                aud = preprocess_audio(audio_file)
                pred = audio_model.predict(aud)[0]
                aud_emotion = AUDIO_CLASSES[np.argmax(pred)]

            if img_emotion:
                st.markdown(f"<div class='result-box'>IMAGE EMOTION<br>{img_emotion}</div>", unsafe_allow_html=True)

            if aud_emotion:
                st.markdown(f"<div class='result-box'>AUDIO EMOTION<br>{aud_emotion}</div>", unsafe_allow_html=True)

# ------------------ ABOUT ------------------
if menu == "About":

    st.title("About PAWMOOD")

    st.markdown("""
    <div class="card">
    <h3>Why PAWMOOD?</h3>
    PAWMOOD was built to help cat owners understand their pets better. Many cats suffer silently — their emotions are hidden in expressions and sounds. This system uses AI to detect those feelings early.

    <h3>Technology</h3>
    • TensorFlow deep learning models  
    • CNN for facial emotion detection  
    • MFCC + Neural Networks for sound emotion  
    • Streamlit web platform  

    <h3>Built by</h3>
    Madhumithaa D K  
    
    </div>
    """, unsafe_allow_html=True)
