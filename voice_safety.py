import numpy as np
import pandas as pd
import librosa
import re
import torch
import joblib
import matplotlib.pyplot as plt

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

data = pd.read_csv('data.csv', encoding='latin-1')

alay_dict = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
alay_dict = alay_dict.rename(columns={0: 'original',
                                      1: 'replacement'})

id_stopword_dict = pd.read_csv('stopwordbahasa.csv', header=None)
id_stopword_dict = id_stopword_dict.rename(columns={0: 'stopword'})

"""Text Data"""

print("Shape: ", data.shape)
data.head(15)

data.HS.value_counts()

data.Abusive.value_counts()

print("Toxic shape: ", data[(data['HS'] == 1) | (data['Abusive'] == 1)].shape)
print("Non-toxic shape: ", data[(data['HS'] == 0) & (data['Abusive'] == 0)].shape)

"""Alay Dictionary"""

print("Shape: ", alay_dict.shape)
alay_dict.head(15)

"""ID Stopword"""

print("Shape: ", id_stopword_dict.shape)
id_stopword_dict.head()

"""Preprocess"""

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def lowercase(text):
    return text.lower()

def remove_unnecessary_char(text):
    text = re.sub('\n',' ',text) # Remove every '\n'
    text = re.sub('rt',' ',text) # Remove every retweet symbol
    text = re.sub('user',' ',text) # Remove every username
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) # Remove every URL
    text = re.sub('  +', ' ', text) # Remove extra spaces
    return text

def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    return text

alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

def remove_stopword(text):
    text = ' '.join(['' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text) # Remove extra spaces
    text = text.strip()
    return text

def stemming(text):
    return stemmer.stem(text)

print("remove_nonaplhanumeric: ", remove_nonaplhanumeric("Halooo,,,,, duniaa!!"))
print("lowercase: ", lowercase("Halooo, duniaa!"))
print("stemming: ", stemming("Perekonomian Indonesia sedang dalam pertumbuhan yang membanggakan"))
print("remove_unnecessary_char: ", remove_unnecessary_char("Hehe\n\n RT USER USER apa kabs www.google.com\n  hehe"))
print("normalize_alay: ", normalize_alay("aamiin adek abis"))
print("remove_stopword: ", remove_stopword("ada hehe adalah huhu yang hehe"))

def preprocess(text):
    text = lowercase(text) # 1
    text = remove_nonaplhanumeric(text) # 2
    text = remove_unnecessary_char(text) # 2
    text = normalize_alay(text) # 3
    text = stemming(text) # 4
    text = remove_stopword(text) # 5
    return text

data['Tweet'] = data['Tweet'].apply(preprocess)

print("Shape: ", data.shape)
data.head(15)

"""# Split, Random Forest, dan Evaluasi"""

# Pilih fitur dan label
X = data['Tweet']  # Fitur (teks)
y = data['HS']  # Label (ubah sesuai target, misalnya 'HS' untuk hate speech)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prediksi
y_pred = rf_model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy:.4f}')
print('Laporan Klasifikasi:')
print(classification_report(y_test, y_pred))

# Asumsikan model sudah dilatih dan prediksi sudah dilakukan
# y_test adalah label asli dari data pengujian
# y_pred adalah hasil prediksi dari model
y_pred = rf_model.predict(X_test)  # Ganti dengan hasil prediksi model Anda

# Menghitung confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Menampilkan confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Tidak Berbahaya', 'Berbahaya'])
disp.plot(cmap=plt.cm.Blues)  # Anda bisa memilih cmap sesuai keinginan

plt.title("Confusion Matrix")
plt.show()

# Simpan model TF-IDF dan Random Forest
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")

print("Model berhasil disimpan!")

# Hitung jumlah dari TP, TN, FP, FN
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# Menyiapkan data untuk visualisasi bar chart
labels = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
values = [TP, TN, FP, FN]

# Membuat bar chart
plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['green', 'blue', 'orange', 'red'])
plt.title('Prediction Results')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

# Menghitung distribusi prediksi untuk kategori 'Berbahaya' dan 'Tidak Berbahaya'
labels = ['Tidak Berbahaya', 'Berbahaya']
sizes = [TN + FP, TP + FN]  # Jumlah prediksi untuk kategori tidak berbahaya dan berbahaya

# Membuat pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])
plt.title('Distribusi Prediksi Model untuk Kategori Berbahaya dan Tidak Berbahaya')
plt.axis('equal')  # Membuat pie chart berbentuk lingkaran
plt.show()

"""# Program

Menggunakan OpenAI/Whisper
"""

# Load model Whisper-Medium dari Hugging Face
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

# Fungsi transkripsi audio ke teks
def transcribe_audio(audio_path):
    # Load dan resample audio ke 16kHz
    audio, _ = librosa.load(audio_path, sr=16000)

    # Preprocessing input
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

    # Paksa Bahasa Indonesia (opsional)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="indonesian", task="transcribe")

    # Transkripsi
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription

# Load model TF-IDF dan Random Forest dari Drive
vectorizer = joblib.load("tfidf_vectorizer.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# Fungsi deteksi ujaran kebencian
def detect_hate_speech(audio_path):
    print("🔄 Memproses audio...")
    transcribed_text = transcribe_audio(audio_path)
    print(f"\n🗣️ Hasil Transkripsi: {transcribed_text}")

    text_vectorized = vectorizer.transform([transcribed_text])
    prediction = rf_model.predict(text_vectorized)[0]

    if prediction == 1:
        print("🚨 Deteksi: Ujaran Berbahaya (Hate Speech)")
    else:
        print("✅ Deteksi: Aman (Tidak Mengandung Ujaran Berbahaya)")

# Contoh penggunaan
audio_path = "windah.wav"
detect_hate_speech(audio_path)
