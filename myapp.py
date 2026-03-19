import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
import os

# --- CONFIG & DARK MODE STYLING ---
MODEL_PATH = "keras_model.h5"

st.set_page_config(page_title="SmartCycle AI | Dark Mode", page_icon="♻️", layout="centered")

# Custom CSS voor een strakke zwarte achtergrond en witte tekst
st.markdown("""
    <style>
    /* De volledige achtergrond zwart */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
        background-image: url("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Recycling_sign_green.png/960px-Recycling_sign_green.png");
        background-repeat: repeat;
        background-size: 50px 50px;
    }

    /* Titels wit maken */
    h1, h2, h3, p {
        color: #FFFFFF !important;
    }

    /* Witte kaarten met donkere randen voor de content */
    .stMarkdown, .stCameraInput, .stFileUploader, .stImage {
        background-color: #161B22;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #30363D;
        margin-bottom: 1rem;
    }

    /* Styling voor de knoppen onderaan */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
        background-color: #21262D;
        color: white;
        border: 1px solid #F0F6FC;
    }

    .stButton>button:hover {
        background-color: #30363D;
        border-color: #8B949E;
    }
    </style>
    """, unsafe_allow_html=True)


# --- Model laden ---
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Modelbestand 'keras_model.h5' niet gevonden!")
        return None, None
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        class_names = ["PMD", "Papier en karton", "GFT", "Glas", "Restafval"]
        return model, class_names
    except Exception as e:
        st.error(f"❌ Fout bij laden model: {e}")
        return None, None


model, class_names = load_my_model()

# Afval details voor de knoppen
bin_details = {
    "PMD": "🔵 **PMD:** Plastic flessen, Metalen verpakkingen en Drinkpakken. Géén harde plastics of vuile folies.",
    "Papier en karton": "📦 **Papier:** Propere kartonnen dozen en papier. Géén behangpapier of vuile pizzadozen.",
    "GFT": "🟢 **GFT:** Etensresten, schillen, fijn tuinafval. Géén vloeistoffen of kattenbakvulling.",
    "Glas": "🍷 **Glas:** Enkel lege glazen flessen en bokalen zonder dop. Géén hittebestendig glas (zoals ovenschalen).",
    "Restafval": "⚪ **Restafval:** Alles wat niet in de andere bakken mag en niet recycleerbaar is."
}


# --- Predictie Functie ---
def predict_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized
    prediction = model.predict(data)[0]
    top_indices = prediction.argsort()[-3:][::-1]
    return [{"class": class_names[i], "confidence": float(prediction[i])} for i in top_indices]


# --- UI APP ---
def run_app():
    st.title("♻️ SmartCycle AI")
    st.write("Detecteer afval in 'Dark Mode'. Snel, strak en duurzaam.")

    # --- SCANNER SECTIE ---
    st.markdown("### 📸 Scan")
    option = st.radio("Invoermethode:", ("Camera 📸", "Upload 🖼️"), horizontal=True)

    image = None
    if option == "Camera 📸":
        img_file = st.camera_input("Maak een foto van het object")
        if img_file: image = Image.open(img_file).convert("RGB")
    else:
        uploaded_file = st.file_uploader("Kies een bestand", type=["jpg", "png", "jpeg"])
        if uploaded_file: image = Image.open(uploaded_file).convert("RGB")

    if image:
        st.image(image, use_container_width=True)
        with st.spinner("⚡ AI analyseert..."):
            results = predict_image(image)

        best = results[0]
        st.success(f"**Gedetecteerd:** {best['class']} ({round(best['confidence'] * 100)}% zekerheid)")

        # Informatie box over het resultaat
        st.info(f"💡 **Sorteeradvies:** {bin_details[best['class']]}")

    st.divider()

    # --- INFO KNOPPEN SECTIE ---
    st.markdown("### ℹ️ Sorteerhulp per categorie")
    st.write("Klik op een knop voor meer informatie over de afvalstroom:")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("PMD 🔵"): st.info(bin_details["PMD"])
    with col2:
        if st.button("Papier 📦"): st.warning(bin_details["Papier en karton"])
    with col3:
        if st.button("GFT 🟢"): st.success(bin_details["GFT"])
    with col4:
        if st.button("Glas 🍷"): st.info(bin_details["Glas"])
    with col5:
        if st.button("Rest ⚪"): st.write(bin_details["Restafval"])

    # Footer in de sidebar
    st.sidebar.title("Project Info")
    st.sidebar.write("Ontwikkeld voor de MoreProf Week.")
    st.sidebar.markdown("---")
    st.sidebar.write("🛠️ **Tech Stack:**")
    st.sidebar.code("Python\nStreamlit\nTensorFlow\nKeras")
