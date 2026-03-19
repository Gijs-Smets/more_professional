import streamlit as st
import tensorflow as tf
import tf_keras as keras
from PIL import Image, ImageOps
import numpy as np
import os

# --- CONFIG ---
MODEL_PATH = "keras_model.h5"

# --- Model laden ---
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Modelbestand niet gevonden!")
        return None, None

    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        class_names = [
            "PMD",
            "Papier en karton",
            "GFT",
            "Glas",
            "Restafval"
        ]
        return model, class_names
    except Exception as e:
        st.error(f"❌ Fout bij laden model: {e}")
        return None, None


model, class_names = load_my_model()

# --- Afval advies ---
bin_advice = {
    "PMD": "Plastic, Metaal, Drinkpakken → PMD-container",
    "Papier en karton": "Papier en karton → Papierbak",
    "GFT": "Groente-, fruit-, tuinafval → GFT-bak",
    "Glas": "Glas → Glasbak",
    "Restafval": "Restafval → Restafvalbak"
}

# --- Predictie ---
def predict_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)[0]

    # Top 3 voorspellingen
    top_indices = prediction.argsort()[-3:][::-1]

    results = []
    for i in top_indices:
        results.append({
            "class": class_names[i],
            "confidence": float(prediction[i])
        })

    return results


# --- UI ---
def run_app():
    st.set_page_config(page_title="♻️ Afval Scanner", layout="centered")

    st.title("♻️ Slimme Afval Scanner")
    st.write("Upload of neem een foto en ontdek waar je afval hoort!")

    option = st.radio(
        "Kies methode:",
        ("📷 Camera", "🖼️ Upload")
    )

    image = None

    if option == "📷 Camera":
        img_file = st.camera_input("Maak een foto")
        if img_file:
            image = Image.open(img_file).convert("RGB")

    else:
        uploaded_file = st.file_uploader("Upload afbeelding", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

    if image:
        st.image(image, caption="📸 Jouw afbeelding", use_container_width=True)

        with st.spinner("🔍 AI is aan het analyseren..."):
            results = predict_image(image)

        st.divider()
        st.subheader("📊 Resultaten")

        # Beste voorspelling
        best = results[0]
        advice = bin_advice.get(best["class"], "Onbekend type afval")

        st.success(f"**Beste match:** {best['class']}")
        st.write(f"**Advies:** {advice}")

        # Confidence bar
        st.progress(int(best["confidence"] * 100))
        st.write(f"Zekerheid: {best['confidence'] * 100:.1f}%")

        st.divider()
        st.subheader("🔎 Top 3 voorspellingen")

        for res in results:
            st.write(f"**{res['class']}**")
            st.progress(int(res["confidence"] * 100))
            st.write(f"{res['confidence'] * 100:.1f}%")
            st.write("---")
