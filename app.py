import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set page title
st.set_page_config(page_title="Skin Lesion Analyzer", layout="centered")

@st.cache_resource
def load_model():
    # Replace with your actual model file path
    model = tf.keras.models.load_model('skin_model.h5')
    return model

def predict(image_data, model):
    size = (224, 224)    
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img_array = np.asarray(image)
    img_reshape = img_array[np.newaxis, ...]
    img_reshape = img_reshape / 255.0 # Normalization
    prediction = model.predict(img_reshape)
    return prediction

model = load_model()

st.title("🩺 Skin Lesion Classification")
st.write("Upload a dermatoscopic image to get an AI-based analysis.")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is not None:
    image = Image.open(file)
    st.image(image, use_container_width=True)
    
    if st.button("Analyze"):
        with st.spinner('Analyzing...'):
            predictions = predict(image, model)
            # Example classes - ensure these match your dataset's order
            classes = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis', 
                       'Dermatofibroma', 'Melanoma', 'Nevus', 'Vascular Lesion']
            
            result = classes[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
            
            st.success(f"Prediction: {result}")
            st.info(f"Confidence Level: {confidence:.2f}%")
