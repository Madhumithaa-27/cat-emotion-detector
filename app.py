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

# ------------------ THEME ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #F8F7BA, #FFF9E6);
}

[data-testid="stSidebar"] {
    background-color: #EFECA3;
}

.card {
    background: white;
    padding: 28px;
    border-radius: 18px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
    margin-bottom: 25px;
}

.step {
    background: #FFF3BF;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 10px;
    border-left: 6px solid #FFB703;
}

.result-box {
    background: linear-gradient(135deg, #D8F3DC, #B7E4C7);
    padding: 30px;
    border-radius: 18px;
    border-left: 8px solid #52B788;
    font-size: 26px;
    font-weight: 800;
    text-align: center;
    color: #1B4332;
    animation: pop 0.4s ease-out;
}

@keyframes pop {
  0% {transform: scale(0.7); opacity: 0;}
  100% {transform: scale(1); opacity: 1;}
}

.stButton>button {
    background: linear-gradient(135deg, #FFB703, #FB8500);
    color: black;
    border-radius: 12px;
    padding: 14px 28px;
    font-size: 16px;
    font-weight: 700;
    border: none;
}
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
menu = st.sidebar.radio("Navigation", ["About", "Predict"])

# ------------------ ABOUT ------------------
if menu == "About":
    st.title("Cat Emotion Detection System")

    st.markdown("""
<div class="card">
This web application uses Artificial Intelligence to understand how a cat is feeling by analyzing images and sounds.
Cats cannot tell us when they are happy, scared, angry, or in pain. Many cat owners struggle to understand their pet’s emotions.
This system was built to help cat owners, veterinarians, and animal lovers better understand cat behavior using technology.
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card">
<h3>How to Use</h3>
<div class="step">Step 1: Upload a clear image of a cat</div>
<div class="step">Step 2: Upload a cat sound (optional)</div>
<div class="step">Step 3: Click Detect Emotion</div>
<div class="step">Step 4: The system will display the detected emotion</div>
</div>
""", unsafe_allow_html=True)

    st.image("https://images.unsplash.com/photo-1592194996308-7b43878e84a6", use_column_width=True)
    st.image("https://images.unsplash.com/photo-1601758123927-1985e4c92d5a", use_column_width=True)

# ------------------ PREDICT ------------------
if menu == "Predict":

    st.title("Detect Cat Emotion")
    st.markdown("<div class='card'>Upload a cat image or sound and click Detect Emotion.</div>", unsafe_allow_html=True)

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

                final_emotion = None

                if image_file:
                    img = preprocess_image(image_file)
                    img_pred = image_model.predict(img)[0]
                    final_emotion = IMAGE_CLASSES[np.argmax(img_pred)]

                if audio_file:
                    aud = preprocess_audio(audio_file)
                    aud_pred = audio_model.predict(aud)[0]
                    final_emotion = AUDIO_CLASSES[np.argmax(aud_pred)]

            st.markdown(f"""
            <div class="result-box">
            {final_emotion}
            </div>
            """, unsafe_allow_html=True)

    st.image("https://images.unsplash.com/photo-1518791841217-8f162f1e1131", use_column_width=True)





