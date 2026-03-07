import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="AI Fruit Classifier", 
    page_icon="🍎", 
    layout="centered"
)

# --- 2. TASARIM VE BAŞLIK ---
st.title("🍎 AI Fruit Classifier")
st.write("Yapay zeka ile meyveleri saniyeler içinde tanıyın. Bir fotoğraf yükleyin ve modelin analizini görün!")
st.divider()

# --- 3. MODEL YÜKLEME (Versiyon Uyumlu Strateji) ---
@st.cache_resource
def load_my_model():
    # Eğitimde kullanılan mimari (100x100 giriş boyutu)
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(100, 100, 3))
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax') 
    ])
    
    # Colab'dan indirip GitHub'a yüklediğin ağırlık dosyası (.h5 uzantılı)
    model.load_weights("fruit_model_weights.weights.h5")
    return model

try:
    model = load_my_model()
except Exception as e:
    st.error("Model yüklenemedi. Lütfen ağırlık dosyasının GitHub'da olduğundan emin olun.")

# --- 4. SINIF İSİMLERİ ---
class_names = [
    'Apple Granny Smith', 'Cauliflower', 'Cherry', 'Cucumber',
    'Mandarine', 'Nut', 'Pear Williams', 'Pepper Yellow',
    'Pineapple', 'Raspberry'
]

# --- 5. DOSYA YÜKLEME VE ANALİZ ---
uploaded_file = st.file_uploader("Bir meyve resmi sürükleyin veya seçin", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Resmi oku
    image = Image.open(uploaded_file).convert("RGB")
    
    # Görsel Düzeni (İki Sütun)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    # Preprocessing (Eğitimle aynı: 100x100)
    image_resized = image.resize((100, 100)) 
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin Süreci
    with st.spinner('Yapay zeka analiz ediyor...'):
        prediction = model.predict(img_array)
        index = np.argmax(prediction)
        predicted_class = class_names[index]
        confidence = prediction[0][index]

    with col2:
        st.subheader("Analiz Sonucu")
        st.success(f"**Tahmin:** {predicted_class}")
        st.metric(label="Accuracy: ", value=f"%{confidence*100:.2f}")
        st.progress(float(confidence))
        
        st.divider()
        
        # --- TOP-3 TAHMİN BÖLÜMÜ ---
        st.write("**En Olası 3 Tahmin:**")
        top3_idx = prediction[0].argsort()[-3:][::-1]
        
        for i in top3_idx:
            score = prediction[0][i] * 100
            st.write(f"**{class_names[i]}**: %{score:.1f}")
            st.progress(float(prediction[0][i]))

    st.divider()

    # --- 6. PROBABILITY CHART (GRAFİKSEL GÖSTERİM) ---
    st.subheader("📊 Tüm Olasılık Dağılımları")
    
    # Grafik verisini DataFrame olarak hazırla
    chart_data = pd.DataFrame(
        prediction[0], 
        index=class_names, 
        columns=["Olasılık"]
    )
    
    # Çubuk grafik
    st.bar_chart(chart_data)

    # Detay Tablosu
    with st.expander("Tüm matematiksel detayları gör"):
        st.table(chart_data)

else:
    st.info("Lütfen analiz için bir resim yükleyin.")
