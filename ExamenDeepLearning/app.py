import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# ✅ Chargement des modèles
@st.cache_resource
def load_cats_dogs_model():
    return load_model("CAT_DOG.h5")

@st.cache_resource
def load_malaria_model():
    return load_model("Cell_img.h5")  # Corrigé ici (plus de dossier models/)

@st.cache_resource
def load_cifar10_model():
    return load_model("CIFAR10_CNN.h5")

# ✅ Prédictions
def predict_cats_dogs(image):
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = cats_dogs_model.predict(img_array)[0][0]
    label = "Chien" if prediction >= 0.5 else "Chat"
    return label, float(prediction)

def predict_malaria(image):
    image = image.resize((50, 50))  # ✅ Taille exacte attendue
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = malaria_model.predict(img_array)[0]
    label = "Parasitée" if np.argmax(prediction) == 1 else "Non parasitée"
    return label, float(np.max(prediction))


def predict_cifar10(image):
    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = cifar10_model.predict(img_array)[0]
    class_index = np.argmax(predictions)
    cifar10_labels = ['avion', 'auto', 'oiseau', 'chat', 'cerf', 
                      'chien', 'grenouille', 'cheval', 'bateau', 'camion']
    return cifar10_labels[class_index], float(predictions[class_index])

# ✅ Interface Streamlit
st.title("🧠 Application de classification d'images par Deep Learning")

model_choice = st.selectbox("Choisissez un modèle :", [
    "Chat vs Chien",
    "Cellules Parasitaires",
    "CIFAR-10"
])

uploaded_file = st.file_uploader("Uploadez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", use_column_width=True)

    with st.spinner("Analyse en cours..."):
        if model_choice == "Chat vs Chien":
            cats_dogs_model = load_cats_dogs_model()
            label, proba = predict_cats_dogs(image)
        elif model_choice == "Cellules Parasitaires":
            malaria_model = load_malaria_model()
            label, proba = predict_malaria(image)
        elif model_choice == "CIFAR-10":
            cifar10_model = load_cifar10_model()
            label, proba = predict_cifar10(image)

    st.success(f"✅ Résultat : {label} ({proba*100:.2f}%)")
else:
    st.info("Veuillez uploader une image pour commencer.")
