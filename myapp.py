import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2

# --- Model laden ---
@st.cache_resource
def load_my_model():
    model = load_model("keras_Model.h5", compile=False)
    class_names = [
        "PMD",
        "Papier en karton",
        "GFT",
        "Glas",
        "Restafval"
    ]
    return model, class_names

model, class_names = load_my_model()

# --- Functie voor voorspelling ---
def predict_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)

    class_name = class_names[index]
    confidence = prediction[0][index]

    # Afval advies
    bin_advice = {
        "PMD": "Plastic, Metaal, Drinkpakken → PMD-container",
        "Papier en karton": "Papier en karton → Papierbak",
        "GFT": "Groente-, fruit-, tuinafval → GFT-bak",
        "Glas": "Glas → Glasbak",
        "Restafval": "Restafval → Restafvalbak"
    }
    advice = bin_advice.get(class_name, "Onbekend type afval")
    return class_name, confidence, advice

# --- Streamlit UI ---
def run_app():
    st.set_page_config(page_title="Afval Classificatie App", layout="centered")
    st.title("♻️ Afval Classificatie App")

    option = st.radio(
        "Kies een optie:",
        ("📷 Foto nemen", "🖼️ Upload")
    )

    image = None

    # Foto nemen
    if option == "📷 Foto nemen":
        img_file = st.camera_input("Neem een foto")

        if img_file:
            image = Image.open(img_file).convert("RGB")
            st.image(image, caption="Genomen foto")

    # Upload
    elif option == "🖼️ Upload":
        uploaded_file = st.file_uploader("Upload afbeelding", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Geüploade foto")

    # Predictie
    if image is not None:
        class_name, confidence, advice = predict_image(image)
        st.subheader("Resultaat:")
        st.write(f"**Afval type:** {class_name}")
        st.write(f"**Confidence:** {confidence:.2f}")
        st.info(f"**Vuilbak advies:** {advice}")