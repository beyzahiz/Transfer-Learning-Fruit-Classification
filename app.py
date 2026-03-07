import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Sayfa başlığı
st.set_page_config(page_title="Meyve Sınıflandırma", page_icon="🍎")
st.title("🍎 Fruit Classification App")

# Modeli yükle
#streamlit normal bir python scripti gibi çalışmaz. dosya yükleyip, bir şeyler değiştikçe streamlit dosyanın tamamını baştan sona tekrar çalıştırır
@st.cache_resource  #fonksiyon çıktısını hafızada tutulup aynı şeyin yeniden hesaplanmasını önler
#ilk çalışmada model yüklenir ram'e alınır sonraki interactionlarda yeniden yüklenmez
def load_my_model():
    # Eğitimde kullandığın mimarinin birebir aynısı
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax') # 10 sınıfın olduğu için 10 yazdık
    ])
    
    # Colab'dan indirip GitHub'a yüklediğin ağırlık dosyası
    model.load_weights("fruit_model_weights.weights.h5")
    return model

# Modeli belleğe al
try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")

# Sınıf isimleri (Eğitimdeki sırayla aynı olduğundan emin ol!)
class_names = [
    'Apple Granny Smith 1', 'Cauliflower 1', 'Cherry 5', 'Cucumber 11',
    'Mandarine 1', 'Nut 4', 'Pear Williams 1', 'Pepper Yellow 1',
    'Pineapple 1', 'Raspberry 2'
]

# 2. ADIM: Dosya Yükleme
uploaded_file = st.file_uploader("Bir meyve resmi yükle", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB") # RGBA hatasını önlemek için RGB'ye çeviriyoruz
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    # 3. ADIM: Preprocessing (Eğitimle birebir aynı!)
    # ÖNEMLİ: Eğitimde 100x100 kullandığın için burası da 100x100 olmalı
    image_resized = image.resize((100, 100)) 
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin
    with st.spinner('Tahmin yapılıyor...'):
        prediction = model.predict(img_array)
        index = np.argmax(prediction)
        predicted_class = class_names[index]
        confidence = prediction[0][index]

    # Sonuçları göster
    st.divider()
    st.subheader(f"Tahmin: **{predicted_class}**")
    st.progress(float(confidence))
    st.write(f"Güven Oranı: %{confidence*100:.2f}")