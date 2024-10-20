import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import io
import requests

# Set page config at the very beginning
st.set_page_config(page_title="Plant Disease Classifier", layout="wide")

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Pesticide_Sprayer.h5')

model = load_model()

# Load disease information
@st.cache_data
def load_disease_info():
    return pd.read_csv('Pesto_Data.csv')

disease_info = load_disease_info()

# Preprocess image
def preprocess_image(image):
    img = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Predict function
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Sidebar navigation
st.sidebar.title("Navigation")

if st.sidebar.button("Home", key="home_button"):
    st.session_state.page = "Home"
st.sidebar.markdown("---")

if st.sidebar.button("Predict Disease", key="predict_disease_button"):
    st.session_state.page = "Predict"
st.sidebar.markdown("---")

if st.sidebar.button("Live Data Capture", key="live_data_capture_button"):
    st.session_state.page = "Live Data Capture"
st.sidebar.markdown("---")

# Uncomment and modify this section if you have user roles
# if 'user' in st.session_state and st.session_state.user['role'] in ['admin', 'doctor']:
#     if st.sidebar.button("Patient Logs", key="patient_logs_button"):
#         st.session_state.page = "Patient Logs"
#     st.sidebar.markdown("---")

if st.sidebar.button("Developers", key="developers_button"):
    st.session_state.page = "Developers"

# Initialize session state for page if not exists
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Main content
if st.session_state.page == "Home":
    st.title("Plant Disease Classification")
    st.write("Welcome to the Plant Disease Classification app!")
    
    # Display some general stats
    st.subheader("General Statistics")
    st.write(f"Number of diseases in database: {len(disease_info)}")
    st.write(f"Model input shape: {model.input_shape}")
    
    # You can add more stats or visualizations here

elif st.session_state.page == "Predict":
    st.title("Disease Prediction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Analyze'):
            # Send the image to Flask backend
            files = {'file': uploaded_file.getvalue()}
            response = requests.post('http://localhost:5000/predict', files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.write(f"Predicted Disease: {result['predicted_disease']}")
                st.write(f"Confidence: {result['confidence']:.2f}")
                
                st.subheader("Disease Information")
                st.write(f"Best Pesticides: {result['best_pesticides']}")
                st.write(f"Worst Pesticides: {result['worst_pesticides']}")
            else:
                st.error("An error occurred during prediction. Please try again.")

elif st.session_state.page == "Live Data Capture":
    st.title("Live Data Capture")
    st.write("This feature is not implemented yet.")

elif st.session_state.page == "Developers":
    st.title("Our Team")
    
    # Custom CSS for developer cards
    st.markdown("""
    <style>
    .dev-card {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .dev-image {
        border-radius: 50%;
        width: 150px;
        height: 150px;
        object-fit: cover;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create three columns for developer cards
    col1, col2, col3 = st.columns(3)
    
    developers = [
        {"name": "Developer 1", "reg_no": "12345", "image": "https://picsum.photos/150/150?random=1"},
        {"name": "Developer 2", "reg_no": "67890", "image": "https://picsum.photos/150/150?random=2"},
        {"name": "Developer 3", "reg_no": "13579", "image": "https://picsum.photos/150/150?random=3"}
    ]
    
    for col, dev in zip([col1, col2, col3], developers):
        with col:
            st.markdown(f"""
            <div class="dev-card">
                <img src="{dev['image']}" class="dev-image">
                <h3>{dev['name']}</h3>
                <p>Reg No: {dev['reg_no']}</p>
            </div>
            """, unsafe_allow_html=True)
