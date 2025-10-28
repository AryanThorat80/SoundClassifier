import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import soundfile as sf
from scipy.signal import resample  # ✅ for resampling

# --- Page Config ---
st.set_page_config(
    page_title="Urban Sound Classifier",
    page_icon="🎧",
    layout="centered"
)

# --- Centered title and description ---
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        color: #4F46E5;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 25px;
    }
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🎧 Urban Sound Classification 🎧</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Identify sounds using your locally stored YAMNet model</div>', unsafe_allow_html=True)

# --- Load YAMNet model and class map ---
@st.cache_resource
def load_model():
    model = tf.saved_model.load("yamnet_model")
    labels_df = pd.read_csv("yamnet_model/yamnet_class_map.csv")
    return model, labels_df

yamnet_model, labels_df = load_model()

# --- Upload audio ---
st.markdown("### 🔊 Upload Audio File")
uploaded_file = st.file_uploader("", type=["wav"])

if uploaded_file is not None:
    wav_data, sr = sf.read(uploaded_file)

    # Convert stereo to mono
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)

    # Resample to 16 kHz if needed
    if sr != 16000:
        st.warning(f"⚠️ Sample rate is {sr} Hz — YAMNet expects 16kHz. Resampling applied.")
        target_len = int(len(wav_data) * 16000 / sr)
        wav_data = resample(wav_data, target_len)
        sr = 16000

    # Audio player
    st.markdown("### ▶️ Listen to Uploaded Audio")
    st.audio(uploaded_file, format="audio/wav")

    # Run prediction
    wav_tensor = tf.convert_to_tensor(wav_data, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(wav_tensor)
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)

    # Get top 5 predictions
    top5_i = np.argsort(mean_scores)[::-1][:5]
    top5_labels = labels_df.loc[top5_i, 'display_name'].values
    top5_scores = mean_scores[top5_i]

    # --- Display results ---
    st.markdown("### 🎯 Top Predictions")
    for label, score in zip(top5_labels, top5_scores):
        st.markdown(f"- **{label}** — {score:.2%}")

    # --- Chart ---
    st.markdown("### 📊 Confidence Chart")
    chart_data = pd.DataFrame({'Label': top5_labels, 'Confidence': top5_scores}).set_index('Label')
    st.bar_chart(chart_data)

else:
    st.info("👆 Please upload a `.wav` file to start classification.")
