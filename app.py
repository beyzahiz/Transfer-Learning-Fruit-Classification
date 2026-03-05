import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Sayfa başlığı
st.title("🍎 Fruit Classification App")

# Modeli yükle
#streamlit normal bir python scripti gibi çalışmaz. dosya yükleyip, bir şeyler değiştikçe streamlit dosyanın tamamını baştan sona tekrar çalıştırır
@st.cache_resource  #fonksiyon çıktısını hafızada tutulup aynı şeyin yeniden hesaplanmasını önler
#ilk çalışmada model yüklenir ram'e alınır sonraki interactionlarda yeniden yüklenmez
def load_model():
    return tf.keras.models.load_model("fruit_model_streamlit.h5", compile=False)

model = load_model()

# Sınıf isimleri
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

# Dosya yükleme
uploaded_file = st.file_uploader("Bir meyve resmi yükle", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # Preprocessing (Eğitimle aynı olmalı!)
    image = image.resize((224,224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Tahmin: {predicted_class}")
    st.write(f"Güven Oranı: %{confidence*100:.2f}")