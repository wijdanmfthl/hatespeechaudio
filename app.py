import gradio as gr
import librosa
import joblib
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

# Load models
processor = AutoProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
rf_model = joblib.load("random_forest_model.pkl")

def detect_hate_speech(audio_path):
    if audio_path is None:
        return "Tidak ada audio.", "Tidak dapat diproses"

    # Load audio
    audio, _ = librosa.load(audio_path, sr=16000)

    # Transkripsi dengan Whisper
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="indonesian", task="transcribe")
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Deteksi ujaran kebencian
    text_vector = vectorizer.transform([transcription])
    prediction = rf_model.predict(text_vector)[0]
    label = "🚨 Ujaran Kebencian" if prediction == 1 else "✅ Aman"

    return transcription, label

# Interface Gradio
interface = gr.Interface(
    fn=detect_hate_speech,
    inputs=gr.Audio(source="upload", type="filepath", label="Unggah atau Rekam Audio"),
    outputs=[
        gr.Textbox(label="Hasil Transkripsi"),
        gr.Textbox(label="Hasil Deteksi Ujaran")
    ],
    title="Deteksi Ujaran Kebencian dalam Audio",
    description="Upload atau rekam audio untuk dideteksi transkripsi dan kandungan ujaran kebencian menggunakan Whisper + Random Forest."
)

if __name__ == "__main__":
    interface.launch()
