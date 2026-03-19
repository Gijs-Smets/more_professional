import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
import os

# --- 1. CONFIG & STYLING (Compact & No-Scroll) ---
MODEL_PATH = "keras_model.h5"

st.set_page_config(page_title="SmartCycle AI", page_icon="♻️", layout="centered")

st.markdown("""
    <style>
    /* Compacte achtergrond */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1c2128 100%);
        color: white;
    }

    /* Verwijder standaard Streamlit padding bovenaan */
    .block-container { 
        padding-top: 1rem; 
        padding-bottom: 0rem; 
    }

    /* Titels compacter */
    h1 { color: #00ff88 !important; margin-bottom: 0rem; text-align: center; font-size: 2rem; }
    h3 { margin-top: 0.5rem; margin-bottom: 0.2rem; font-size: 1.1rem; }

    /* De Pop-up stijl (Compact) */
    .result-popup {
        background-color: #161b22;
        padding: 15px;
        border-radius: 12px;
        border-left: 8px solid #00ff88;
        box-shadow: 0 4px 15px rgba(0,255,136,0.2);
        margin: 10px 0;
    }

    /* Knoppen compacter */
    .stButton>button {
        height: 2.2em;
        border-radius: 8px;
        border: 1px solid #00ff88;
        background-color: rgba(0, 255, 136, 0.05);
        color: white;
        font-size: 0.8rem;
    }

    /* Verklein de afstand tussen elementen */
    .stDivider { margin: 0.5rem 0 !important; }

    /* Compacte radio-knop layout */
    div.row-widget.stRadio > div{
        flex-direction:row;
        gap: 10px;
    }

    /* Compacte camera/uploader input */
    .stCameraInput, .stFileUploader {
        border-radius: 10px;
        padding: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 2. MODEL LADEN ---
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        class_names = ["PMD", "Papier en karton", "GFT", "Glas", "Restafval"]
        return model, class_names
    except:
        return None, None


model, class_names = load_my_model()

bin_details = {
    "PMD": "🔵 **PMD:** Plastic flessen, Metalen verpakkingen en Drinkpakken.",
    "Papier en karton": "📦 **Papier:** Propere kartonnen dozen en papier.",
    "GFT": "🟢 **GFT:** Etensresten, schillen, fijn tuinafval.",
    "Glas": "🍷 **Glas:** Enkel lege glazen flessen en bokalen zonder dop.",
    "Restafval": "⚪ **Restafval:** Alles wat niet in de andere bakken mag en niet recycleerbaar is."
}


# --- 3. PREDICTIE ---
def predict_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = (np.asarray(image).astype(np.float32) / 127.5) - 1
    data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
    data[0] = img_array
    prediction = model.predict(data, verbose=0)[0]
    idx = np.argmax(prediction)
    return class_names[idx], float(prediction[idx])


# --- 4. INTERFACE ---
def run_app():
    st.write("<h1>♻️ SmartCycle AI</h1>", unsafe_allow_html=True)

    # INFO SECTIE (Bovenaan, Compact)
    st.markdown("### Sorteerhulp")
    cols = st.columns(5)
    btns = [("PMD 🔵", "PMD"), ("Papier 📦", "Papier en karton"), ("GFT 🟢", "GFT"), ("Glas 🍷", "Glas"),
            ("Rest ⚪", "Restafval")]

    for i, (txt, key) in enumerate(btns):
        with cols[i]:
            if st.button(txt):
                st.info(bin_details[key])

    # Placeholder voor resultaat (Direct onder de knoppen)
    res_place = st.empty()

    st.divider()

    # SCANNER (Compact)
    option = st.radio("Bron:", ("Camera 📸", "Upload 🖼️"), horizontal=True, label_visibility="collapsed")
    img_file = st.camera_input("Scan") if option == "Camera 📸" else st.file_uploader("Foto",
                                                                                     type=["jpg", "png", "jpeg"])

    if img_file:
        try:
            image = Image.open(img_file).convert("RGB")
            st.image(image, use_container_width=True)

            with st.spinner("Analyse..."):
                label, confidence = predict_image(image)
                conf_pct = round(confidence * 100)

            # De gevraagde pop-up tekst, op de perfecte plek
            res_place.markdown(f"""
                <div class="result-popup">
                    <h3 style="margin:0; color:#00ff88;">Gedetecteerd: {label} ({conf_pct}% zekerheid)</h3>
                    <p style="margin-top:5px; color:white;">💡 <b>Sorteeradvies:</b> {bin_details[label]}</p>
                </div>
            """, unsafe_allow_html=True)

        except Exception:
            st.error("Bestand kon niet worden gelezen.")


if __name__ == "__main__":
    if model:
        run_app()
    else:
        st.error("Model 'keras_model.h5' niet gevonden.")
