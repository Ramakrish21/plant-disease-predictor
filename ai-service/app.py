# In ai-service/app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import json

# --- 1. CONFIGURATION AND SETUP ---
# Set the page configuration for a professional look
st.set_page_config(
    page_title="Potato Disease Detector",
    page_icon="🥔",
    layout="centered",
    initial_sidebar_state="auto",
)

# Define paths based on your project structure
# We assume this script is in 'ai-service/', so we go up one level then into 'ai-model'
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'ai-model', 'Model')
MODEL_PATH = os.path.join(MODEL_DIR, 'C:\Users\AU\Downloads\plant-disease-predictor\ai-service\ai-model\Model\image_based_trained_model.keras') # Use the best model saved by your callback
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.json')
IMAGE_SIZE = 224

# --- 2. CACHED HELPER FUNCTIONS ---
# Why we use @st.cache_resource:
# This is a special Streamlit command. It tells Streamlit to load the heavy model only ONCE
# and keep it in memory. This makes the app much faster.
@st.cache_resource
def load_trained_model():
    """Loads and caches the trained Keras model from disk."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}", icon="🚨")
        return None

@st.cache_data
def load_class_names():
    """Loads and caches the class names from the JSON file."""
    try:
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
        print("Class names loaded successfully.")
        return class_names
    except Exception as e:
        st.error(f"Error loading class names: {e}", icon="🚨")
        return None

# Load the model and class names at the start
model = load_trained_model()
class_names = load_class_names()

# --- 3. THE USER INTERFACE ---
# st.title creates the main title of the web page
st.title("🥔 Potato Leaf Disease Detector")

# st.write adds a paragraph of text with markdown support
st.markdown("Upload a clear photo of a potato leaf for an instant diagnosis and severity assessment.")

# st.file_uploader creates the interactive file upload widget
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

# --- 4. PREDICTION LOGIC ---
# This block of code only runs if the model is loaded and a user has uploaded a file.
if uploaded_file is not None and model is not None and class_names is not None:
    
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Your Uploaded Image', use_column_width=True)
    
    # Show a spinner while the model makes a prediction
    with st.spinner('Analyzing the leaf... 🧐'):
        # Preprocess the image to match the model's input requirements
        # 1. Resize the image
        image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        # 2. Convert to numpy array
        img_array = tf.keras.utils.img_to_array(image_resized)
        # 3. Add a batch dimension
        img_array = tf.expand_dims(img_array, 0)

        # Make the prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        # Get the final result
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

    # Display the final prediction in a clean format
    st.divider()
    st.header("Diagnosis Results")
    st.success(f"**Predicted Disease:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # --- 5. ACTIONABLE RECOMMENDATION FEATURE ---
    st.header("Recommended Action")
    if "Severe" in predicted_class:
        st.error("""
        **High Alert:** This appears to be a **severe infection**.
        - **Action:** Immediate removal of the infected plant is recommended to prevent the disease from spreading.
        - **Treatment:** Apply a systemic fungicide to the surrounding plants as a preventive measure.
        - **Next Steps:** Consult a local agricultural expert for aggressive treatment options.
        """, icon="⚠️")
    elif "Mild" in predicted_class:
        st.warning("""
        **Caution:** This appears to be a **mild infection**.
        - **Action:** Isolate the plant if possible and monitor it closely.
        - **Treatment:** Apply a suitable organic (like neem oil) or chemical spray.
        - **Next Steps:** Ensure proper air circulation and avoid over-watering.
        """, icon="🔎")
    elif "Healthy" in predicted_class:
        st.success("""
        **All Clear:** The leaf appears to be **healthy**.
        - **Action:** No immediate action is required.
        - **Next Steps:** Continue with good agricultural practices, regular monitoring, and proper nutrition to maintain plant health.
        """, icon="✅")