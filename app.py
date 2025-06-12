import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import os
from faster_whisper import WhisperModel
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import tempfile
import warnings
warnings.filterwarnings("ignore")

# Konfigurasi halaman
st.set_page_config(
    page_title="Voice Safety - Hate Speech Detection",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .danger-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .safe-box {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .transcription-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        font-family: 'Georgia', serif;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px dashed #cccccc;
        text-align: center;
    }
    .process-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Cache untuk loading model
@st.cache_resource
def load_hate_speech_models():
    """Load model untuk deteksi hate speech"""
    
    with st.spinner("🔄 Loading hate speech detection models..."):
        try:
            # Cek keberadaan file
            required_files = [
                "tfidf_vectorizer.pkl",
                "random_forest_model.pkl", 
                "new_kamusalay.csv",
                "stopwordbahasa.csv"
            ]
            
            missing_files = [f for f in required_files if not os.path.exists(f)]
            if missing_files:
                st.error(f"❌ File tidak ditemukan: {', '.join(missing_files)}")
                return None
            
            # Load TF-IDF dan Random Forest
            vectorizer = joblib.load("tfidf_vectorizer.pkl")
            rf_model = joblib.load("random_forest_model.pkl")
            
            # Load Alay Dictionary
            alay_dict = pd.read_csv("new_kamusalay.csv", encoding='latin-1', header=None)
            alay_dict = alay_dict.rename(columns={0: 'original', 1: 'replacement'})
            alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
            
            # Load Stopwords
            id_stopword_dict = pd.read_csv("stopwordbahasa.csv", header=None)
            id_stopword_dict = id_stopword_dict.rename(columns={0: 'stopword'})
            
            # Initialize Stemmer
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            
            st.success("✅ Hate speech models loaded successfully!")
            
            return {
                'vectorizer': vectorizer,
                'rf_model': rf_model,
                'alay_dict_map': alay_dict_map,
                'stopwords': id_stopword_dict,
                'stemmer': stemmer
            }
            
        except Exception as e:
            st.error(f"❌ Error loading hate speech models: {str(e)}")
            return None

@st.cache_resource
def load_whisper_model():
    """Load Faster Whisper Small model"""
    try:
        with st.spinner("🔄 Loading Faster Whisper Small model..."):
            # Load model dengan device auto-detect
            model = WhisperModel("small", device="cpu", compute_type="int8")
            st.success("✅ Faster Whisper Small model loaded!")
            return model
    except Exception as e:
        st.error(f"❌ Error loading Faster Whisper: {str(e)}")
        return None

def preprocess_text(text, models_dict):
    """Preprocessing text untuk deteksi hate speech"""
    
    if models_dict is None:
        return ""
    
    alay_dict_map = models_dict['alay_dict_map']
    stopwords = models_dict['stopwords']
    stemmer = models_dict['stemmer']
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove non-alphanumeric
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    
    # Step 3: Remove unnecessary chars
    text = re.sub('\n',' ',text)
    text = re.sub('rt',' ',text)
    text = re.sub('user',' ',text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text)
    text = re.sub(' +', ' ', text)
    
    # Step 4: Normalize alay
    text = ' '.join([alay_dict_map.get(word, word) for word in text.split(' ')])
    
    # Step 5: Stemming
    text = stemmer.stem(text)
    
    # Step 6: Remove stopwords
    text = ' '.join(['' if word in stopwords.stopword.values else word for word in text.split(' ')])
    text = re.sub(' +', ' ', text)
    text = text.strip()
    
    return text

def transcribe_audio(audio_path, whisper_model):
    """Transkripsi audio menggunakan Faster Whisper"""
    try:
        # Transkripsi dengan language Indonesian
        segments, info = whisper_model.transcribe(
            audio_path, 
            language="id",  # Indonesian
            beam_size=5,
            word_timestamps=False
        )
        
        # Gabungkan semua segmen
        transcription = ""
        for segment in segments:
            transcription += segment.text + " "
        
        return transcription.strip()
        
    except Exception as e:
        st.error(f"❌ Error in transcription: {str(e)}")
        return ""

def detect_hate_speech(transcribed_text, models_dict, threshold=0.3):
    """Deteksi hate speech"""
    
    if models_dict is None:
        return 0, [0.5, 0.5]
    
    vectorizer = models_dict['vectorizer']
    rf_model = models_dict['rf_model']
    
    # Preprocess
    processed_text = preprocess_text(transcribed_text, models_dict)
    
    if not processed_text.strip():
        return 0, [0.9, 0.1]
    
    # Vectorize dan prediksi
    try:
        text_vectorized = vectorizer.transform([processed_text])
        probabilities = rf_model.predict_proba(text_vectorized)[0]
        
        # Gunakan threshold
        prediction = 1 if probabilities[1] > threshold else 0
        
        return prediction, probabilities
        
    except Exception as e:
        st.error(f"❌ Error in hate speech detection: {str(e)}")
        return 0, [0.5, 0.5]

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Voice Safety</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deteksi Ujaran Kebencian dalam Audio Bahasa Indonesia</p>', unsafe_allow_html=True)
    
    # Load models
    hate_speech_models = load_hate_speech_models()
    whisper_model = load_whisper_model()
    
    if hate_speech_models is None or whisper_model is None:
        st.error("❌ Gagal memuat models. Pastikan semua file tersedia!")
        st.stop()
    
    # Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload File Audio")
    st.markdown("*Pilih file audio untuk dianalisis (WAV, MP3, M4A)*")
    
    uploaded_file = st.file_uploader(
        "",
        type=['wav', 'mp3', 'm4a'],
        help="Upload file audio untuk dianalisis"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Jika file diupload, langsung tampilkan audio player
    if uploaded_file is not None:
        st.markdown("### 🎵 Audio yang Diupload")
        st.audio(uploaded_file, format='audio/wav')
        
        # Process Section
        st.markdown('<div class="process-section">', unsafe_allow_html=True)
        st.markdown("### 🔍 Analisis Audio")
        st.markdown("*Klik tombol di bawah untuk memulai proses transkripsi dan deteksi ujaran kebencian*")
        
        if st.button("🚀 Proses Transkripsi & Deteksi", type="primary"):
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Transkripsi
                status_text.text("🗣️ Sedang melakukan transkripsi...")
                progress_bar.progress(30)
                
                transcription = transcribe_audio(tmp_file_path, whisper_model)
                
                if transcription:
                    progress_bar.progress(70)
                    status_text.text("🔍 Menganalisis konten...")
                    
                    # Step 2: Deteksi hate speech
                    prediction, probabilities = detect_hate_speech(transcription, hate_speech_models)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analisis selesai!")
                    
                    # Hapus progress bar setelah selesai
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Tampilkan hasil transkripsi
                    st.markdown("### 📝 Hasil Transkripsi")
                    st.markdown(f'''
                    <div class="transcription-box">
                        "{transcription}"
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Tampilkan hasil deteksi
                    st.markdown("### 🚨 Hasil Deteksi")
                    
                    if prediction == 1:
                        st.markdown(f'''
                        <div class="danger-box">
                            <h3>🚨 TERDETEKSI UJARAN KEBENCIAN</h3>
                            <p><strong>Tingkat Kepercayaan:</strong> {probabilities[1]:.1%}</p>
                            <p><strong>Status:</strong> Audio mengandung konten yang berpotensi berbahaya</p>
                            <p><strong>Rekomendasi:</strong> Konten ini memerlukan review lebih lanjut</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="safe-box">
                            <h3>✅ KONTEN AMAN</h3>
                            <p><strong>Tingkat Kepercayaan:</strong> {probabilities[0]:.1%}</p>
                            <p><strong>Status:</strong> Audio tidak mengandung ujaran kebencian</p>
                            <p><strong>Rekomendasi:</strong> Konten aman untuk didistribusikan</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Tampilkan distribusi probabilitas
                    st.markdown("### 📊 Detail Analisis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="🟢 Probabilitas Aman",
                            value=f"{probabilities[0]:.1%}",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            label="🔴 Probabilitas Berbahaya", 
                            value=f"{probabilities[1]:.1%}",
                            delta=None
                        )
                    
                    # Progress bars untuk visualisasi
                    st.markdown("**Distribusi Probabilitas:**")
                    st.progress(probabilities[0], text=f"Aman: {probabilities[0]:.1%}")
                    st.progress(probabilities[1], text=f"Berbahaya: {probabilities[1]:.1%}")
                    
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ Gagal melakukan transkripsi audio. Pastikan file audio valid.")
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Placeholder ketika belum ada file
        st.markdown("### 💡 Cara Menggunakan")
        st.markdown("""
        1. **Upload Audio** - Pilih file audio (WAV, MP3, atau M4A)
        2. **Preview Audio** - Audio akan langsung dapat diputar setelah upload
        3. **Proses Analisis** - Klik tombol untuk memulai transkripsi dan deteksi
        4. **Lihat Hasil** - Dapatkan hasil transkripsi dan status keamanan konten
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("### ℹ️ Informasi Teknis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Model yang Digunakan:**")
        st.markdown("- **STT**: Faster Whisper Small")
        st.markdown("- **Classifier**: Random Forest")
        st.markdown("- **Language**: Bahasa Indonesia")
    
    with col2:
        st.markdown("**📈 Performa Model:**")
        st.markdown("- **Akurasi**: ~84%")
        st.markdown("- **Threshold**: 0.3")
        st.markdown("- **Preprocessing**: PySastrawi")
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    **⚠️ Disclaimer:** 
    Aplikasi ini menggunakan model machine learning untuk deteksi ujaran kebencian. 
    Hasil deteksi mungkin tidak 100% akurat dan sebaiknya dikombinasikan dengan review manual 
    untuk konten yang sensitif.
    """)

if __name__ == "__main__":
    main()
