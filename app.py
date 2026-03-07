import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="AI Fruit Classifier", 
    page_icon="🍎", 
    layout="centered"
)

# --- TASARIM VE BAŞLIK ---
st.title("🍎 AI Fruit Classifier")
st.write("Yapay zeka ile meyveleri saniyeler içinde tanıyın. Bir fotoğraf yükleyin ve sonucu görün!")
st.divider()

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_my_model():
    # Eğitimde kullandığın boyut 100x100 ise (100, 100, 3) yapmalısın!
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(100, 100, 3))
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax') 
    ])
    
    # Dosya adının birebir aynı olduğundan emin ol
    model.load_weights("fruit_model_weights.weights.h5")
    return model

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model yüklenirken bir sorun oluştu. Lütfen dosya adını kontrol edin.")

# --- SINIF İSİMLERİ ---
class_names = [
    'Apple Granny Smith', 'Cauliflower', 'Cherry', 'Cucumber',
    'Mandarine', 'Nut', 'Pear Williams', 'Pepper Yellow',
    'Pineapple', 'Raspberry'
]

# --- DOSYA YÜKLEME VE TAHMİN ---
uploaded_file = st.file_uploader("Bir meyve resmi sürükleyin veya seçin", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Resmi oku ve göster
    image = Image.open(uploaded_file).convert("RGB")
    
    # İki sütunlu yapı (Sol: Resim, Sağ: Tahmin Özeti)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    # Preprocessing
    # Eğitim boyutun 100x100 ise burayı 100, 100 bırak
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
        st.metric(label="Güven Oranı", value=f"%{confidence*100:.2f}")
        st.progress(float(confidence))

    st.divider()

    # --- PROBABILITY CHART (PROFESYONEL GRAFİK) ---
    st.subheader("📊 Sınıf Bazlı Olasılık Dağılımı")
    
    # Grafik verisini hazırla
    chart_data = pd.DataFrame(
        prediction[0], 
        index=class_names, 
        columns=["Olasılık"]
    )
    
    # Çubuk grafik oluştur
    st.bar_chart(chart_data)

    # Detaylı Tablo (Opsiyonel)
    with st.expander("Tüm olasılık detaylarını gör"):
        st.table(chart_data)

else:
    st.info("Lütfen analiz için bir resim yükleyin.")

# --- FOOTER ---
st.markdown("---")
st.caption("Geliştirici: Beyza | Transfer Learning Project")