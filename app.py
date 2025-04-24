import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.optimizers import Adamax
import datetime

# === Configuration de la page
st.set_page_config(page_title="Prédiction Cellule Sanguine", layout="centered")
st.title("🩸 Prédiction Cellulaire (CNN)")
st.markdown("Upload une image de cellule sanguine 👇")

# === Charger le modèle
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Bloods.h5", compile=False)
    model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

# === Classes connues (à adapter selon ton modèle)
class_labels = ['basophil', 'erythroblast', 'monocyte', 'myeloblast', 'seg_neutrophil']

# === Historique en session
if "history" not in st.session_state:
    st.session_state.history = []

# === Upload d’image
uploaded_file = st.file_uploader("📁 Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image uploadée", use_container_width=True)

    # === Prétraitement image
    img = image.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)

    # === Prédiction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]).numpy()
    predicted_class = class_labels[np.argmax(score)]
    confidence = 100 * np.max(score)

    # === Top 3 classes
    top_3_idx = np.argsort(score)[-3:][::-1]
    top_3_classes = [(class_labels[i], 100 * score[i]) for i in top_3_idx]

    # === Affichage principal
    st.markdown(f"### ✅ Prédiction : `{predicted_class}` ({confidence:.2f}%)")

    # === Graphique
    st.subheader("📊 Probabilités par classe")
    df_probs = pd.DataFrame({'Classe': class_labels, 'Probabilité (%)': 100 * score})
    st.bar_chart(df_probs.set_index("Classe"))

    # === Affichage top 3
    st.markdown("#### 🥇 Top 3 prédictions")
    for label, prob in top_3_classes:
        st.write(f"- **{label}** : {prob:.2f}%")

    # ✅ Historique compatible avec affichage
    top_3_str = ", ".join([f"{lbl} ({prob:.1f}%)" for lbl, prob in top_3_classes])

    st.session_state.history.append({
      "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      "prédiction": predicted_class,
      "confiance": f"{confidence:.2f}%",
      "top_3": top_3_str
        })

    # === Téléchargement CSV
    st.download_button("⬇️ Télécharger les résultats", df_probs.to_csv(index=False), "prediction_result.csv", "text/csv")

# === Historique global
if st.session_state.history:
    st.subheader("📁 Historique des prédictions")
    st.dataframe(pd.DataFrame(st.session_state.history))
