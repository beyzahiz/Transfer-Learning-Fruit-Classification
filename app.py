import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="AI Fruit Classifier", 
    page_icon="🍎", 
    layout="centered"
)

# --- 2. TASARIM VE BAŞLIK ---
st.title("🍎 AI Fruit Classifier")
st.write("Yapay zeka ile meyveleri saniyeler içinde tanıyın. Bir fotoğraf yükleyin ve modelin analizini görün.")
st.divider()

# --- 3. MODEL YÜKLEME ---
@st.cache_resource
def load_my_model():
    # HEDEF BOYUT: Colab'daki gibi 224x224
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax') 
    ])
    
    # Ağırlıkları yükle
    model.load_weights("fruit_model_weights.weights.h5")
    return model

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model yüklenemedi: {e}")

# --- 4. SINIF İSİMLERİ (Colab ile Birebir Aynı Sıra) ---
class_names = [
    'Apple Granny Smith 1', 
    'Cauliflower 1', 
    'Cherry 5', 
    'Cucumber 11', 
    'Mandarine 1', 
    'Nut 4', 
    'Pear Williams 1', 
    'Pepper Yellow 1', 
    'Pineapple 1', 
    'Raspberry 2'
]

# --- 5. DOSYA YÜKLEME VE ANALİZ ---
uploaded_file = st.file_uploader("Bir meyve resmi sürükleyin veya seçin", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    # Preprocessing
    image_resized = image.resize((224, 224)) 
    img_array = np.array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    # NOT: Eğer tahminler hala yanlışsa bu satırı img_array = img_array / 255.0 ile değiştirip dene.
    img_array = preprocess_input(img_array)

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
        st.write("**En Olası 3 Tahmin:**")
        top3_idx = prediction[0].argsort()[-3:][::-1]
        
        for i in top3_idx:
            score = prediction[0][i] * 100
            st.write(f"**{class_names[i]}**: %{score:.1f}")
            st.progress(float(prediction[0][i]))

    st.divider()
    
    # --- 6. GRAFİK VE MATEMATİKSEL DETAYLAR ---
    st.subheader("📊 Tüm Olasılık Dağılımları")
    chart_data = pd.DataFrame(prediction[0], index=class_names, columns=["Olasılık"])
    
    st.bar_chart(chart_data)

    # Detay Tablosu (Geri eklenen kısım)
    with st.expander("Tüm matematiksel detayları gör"):
        st.write("Her sınıf için hesaplanan ham olasılık değerleri:")
        st.table(chart_data)

else:
    st.info("Lütfen analiz için bir resim yükleyin.")
