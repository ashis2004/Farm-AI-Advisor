import streamlit as st
import numpy as np
import pandas as pd
import requests
import pickle
import io
import torch

from PIL import Image
from torchvision import transforms

from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from utils.model import ResNet9
import config

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Farm AI Advisor",
    page_icon="🌱",
    layout="centered"
)

st.title("🌾 Farm AI Advisor")
st.caption("Crop Recommendation • Fertilizer Suggestion • Disease Detection")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    disease_classes = [
        'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust',
        'Apple___healthy','Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy','Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy','Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot','Peach___healthy',
        'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
        'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
        'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch','Strawberry___healthy',
        'Tomato___Bacterial_spot','Tomato___Early_blight',
        'Tomato___Late_blight','Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus','Tomato___healthy'
    ]

    disease_model = ResNet9(3, len(disease_classes))
    disease_model.load_state_dict(
        torch.load("models/plant_disease_model.pth", map_location="cpu")
    )
    disease_model.eval()

    crop_model = pickle.load(open("models/RandomForest.pkl", "rb"))

    return disease_model, crop_model, disease_classes

disease_model, crop_model, disease_classes = load_models()

# ---------------- UTILS ----------------
def weather_fetch(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={config.weather_api_key}"
    res = requests.get(url).json()
    if res.get("cod") != "404":
        return round(res["main"]["temp"] - 273.15, 2), res["main"]["humidity"]
    return None

def predict_image(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(img))
    tensor = transform(image).unsqueeze(0)
    output = disease_model(tensor)
    _, idx = torch.max(output, 1)
    return disease_classes[idx.item()]

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🌱 Crop", "🧪 Fertilizer", "🍃 Disease"])

# ---------- CROP ----------
with tab1:
    st.subheader("Crop Recommendation")

    N = st.number_input("Nitrogen", 0, 200)
    P = st.number_input("Phosphorous", 0, 200)
    K = st.number_input("Potassium", 0, 200)
    ph = st.number_input("pH", 0.0, 14.0)
    rainfall = st.number_input("Rainfall (mm)")
    city = st.text_input("City")

    if st.button("Predict Crop"):
        weather = weather_fetch(city)
        if weather:
            temp, humidity = weather
            data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
            result = crop_model.predict(data)[0]
            st.success(f"Recommended Crop: 🌾 {result}")
        else:
            st.error("Invalid city name")

# ---------- FERTILIZER ----------
with tab2:
    st.subheader("Fertilizer Recommendation")

    crop = st.text_input("Crop Name")
    N = st.number_input("N", 0, 200)
    P = st.number_input("P", 0, 200)
    K = st.number_input("K", 0, 200)

    if st.button("Recommend Fertilizer"):
        df = pd.read_csv("Data/fertilizer.csv")
        row = df[df['Crop'] == crop].iloc[0]

        diff = {
            abs(row['N'] - N): 'N',
            abs(row['P'] - P): 'P',
            abs(row['K'] - K): 'K'
        }

        key = diff[max(diff.keys())]
        st.info(fertilizer_dic[key + 'High'])

# ---------- DISEASE ----------
with tab3:
    st.subheader("Plant Disease Detection")

    file = st.file_uploader("Upload leaf image", type=["jpg", "png"])

    if file:
        st.image(file, width=250)
        prediction = predict_image(file.read())
        st.warning(disease_dic[prediction])

