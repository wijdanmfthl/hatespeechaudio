import streamlit as st
import tempfile
import librosa
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

st.set_page_config(page_title="Deteksi Ujaran Kebencian Audio", layout="centered")

# Load model Whisper
@st.cache_resource
def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
    return processor, model

# Load model ML
@st.cache_resource
def load_ml_model():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    rf_model = joblib.load("random_forest_model.pkl")
    return vectorizer, rf_model

processor, model = load_whisper_model()
vectorizer, rf_model = load_ml_model()

# Transkripsi
def transcribe_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="indonesian", task="transcribe")
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# Deteksi
def detect_hate_speech_from_audio(audio_path):
    transcription = transcribe_audio(audio_path)
    text_vectorized = vectorizer.transform([transcription])
    prediction = rf_model.predict(text_vectorized)[0]
    return transcription, prediction

# Highlight kata kasar
def highlight_hate_speech(text):
    # Daftar kata kasar sederhana (bisa dikembangkan)
    hate_keywords = ["bodoh", "goblok", "babi", "anjing", "brengsek", "bangsat"]
    for word in hate_keywords:
        text = text.replace(word, f"<span style='color:red;font-weight:bold'>{word}</span>")
    return text

# Session State
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "transcription" not in st.session_state:
    st.session_state.transcription = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# Judul
st.title("Deteksi Ujaran Berbahaya dalam Audio Menggunakan Random Forest dan Whisper")

# Upload Audio
uploaded_file = st.file_uploader("Upload file audio (mp3, wav)", type=["wav", "mp3"])
if uploaded_file:
    st.session_state.audio_bytes = uploaded_file.read()
    st.success("✅ Audio berhasil diupload!")

# Jika audio tersedia, tampilkan player dan tombol deteksi
if st.session_state.audio_bytes:
    st.subheader("🔈 Putar Audio:")
    st.audio(st.session_state.audio_bytes, format="audio/wav")

    if st.button("🚀 Deteksi Sekarang"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(st.session_state.audio_bytes)
            tmp_path = tmp.name

        with st.spinner("🔁 Menganalisis audio..."):
            try:
                transcription, prediction = detect_hate_speech_from_audio(tmp_path)
                st.session_state.transcription = transcription
                st.session_state.prediction = prediction
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat memproses audio: {e}")
            finally:
                os.remove(tmp_path)

# Menampilkan hasil jika sudah dianalisis
if st.session_state.transcription:
    st.subheader("📝 Hasil Transkripsi:")
    highlighted = highlight_hate_speech(st.session_state.transcription.lower())
    st.markdown(f"<div style='background-color:#f8f8f8; padding:10px; border-radius:5px'>{highlighted}</div>", unsafe_allow_html=True)

    st.subheader("🚨 Hasil Deteksi:")
    if st.session_state.prediction == 1:
        st.error("Terdeteksi mengandung Ujaran Kebencian.")
    else:
        st.success("Aman, tidak mengandung ujaran kebencian.")

# Tombol reset
if st.session_state.audio_bytes or st.session_state.transcription:
    if st.button("🔄 Reset"):
        st.session_state.audio_bytes = None
        st.session_state.transcription = None
        st.session_state.prediction = None
        st.experimental_rerun()
