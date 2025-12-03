import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd

# --- Configuration and Setup ---

CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
IMG_SIZE = (224, 224)
MODEL_PATH = "brain_tumor_predictor.keras"

st.set_page_config(
    page_title="🧠 Brain Tumor MRI Classifier",
    page_icon="🔬",
    layout="centered"
)

# --- 🎨 Custom CSS Injection ---
st.markdown(
    """
    <style>
    /* 1. Main Background and Font */
    .stApp {
        background-color: #f7f9fc; /* Light subtle grey background */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* 2. Hide Streamlit Menu and Footer (for a cleaner look) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 3. Title Styling */
    h1 {
        color: #1f78b4; /* Dark blue color */
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px #ccc;
    }
    
    /* 4. Custom Styling for Containers/Bordered Elements (Adds depth) */
    .stContainer {
        border-radius: 10px;
        box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-top: 15px;
    }

    /* 5. Primary Button Styling (Makes it stand out) */
    .stButton>button {
        color: white;
        background-color: #33a02c; /* Green for GO/Classify */
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1em;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d7e21; /* Darker green on hover */
    }

    /* 6. File Uploader Styling */
    .stFileUploader {
        border: 2px dashed #a6cee3;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }

    /* 7. Success/Warning Boxes */
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# --- Model Loading (Cached) ---

@st.cache_resource
def load_keras_model(path):
    """Loads the pre-trained Keras model."""
    try:
        with st.spinner(f"Loading Keras model from {path}..."):
            model = load_model(path)
            st.success("Model loaded successfully!")
            return model
    except Exception as e:
        st.error(f"Error loading model: Could not find or load model at '{path}'. Details: {e}")
        st.stop()

MODEL = load_keras_model(MODEL_PATH)

# --- Utility Functions ---

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resizes and converts the image for model inference."""
    image = image.resize(IMG_SIZE)
    img_array = np.asarray(image)

    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

def predict_and_display(image: Image.Image, model: tf.keras.Model):
    """Handles image preprocessing, prediction, and result display."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text('1/2. Preprocessing image...')
    progress_bar.progress(50)
    
    processed_image = preprocess_image(image)
    
    status_text.text('2/2. Running model inference...')
    progress_bar.progress(75)

    predictions = model.predict(processed_image)
    
    status_text.text('Inference complete.')
    progress_bar.progress(100)
    
    # Prediction Result Container
    with st.container(border=True):
        st.subheader("Classification Result")
        
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100

        if "No Tumor" in predicted_class_name:
            st.balloons()
            st.success(f"🎉 **Predicted Class: {predicted_class_name}**")
        else:
            # Use a dark yellow warning for tumor presence
            st.markdown(f'<div style="background-color: #ff9900; color: white; padding: 10px; border-radius: 5px; font-weight: bold;">⚠️ Predicted Class: {predicted_class_name}</div>', unsafe_allow_html=True)
            
        st.markdown(f"Confidence: **{confidence:.2f}%**")
        
    st.markdown("---")
    
    # Probability Breakdown
    st.subheader("Prediction Probabilities Breakdown")
    
    prob_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Probability': [p * 100 for p in predictions[0]]
    })
    
    prob_df = prob_df.sort_values(by='Probability', ascending=False)
    
    col_chart, col_data = st.columns([3, 1])

    with col_chart:
        # Using the classic Streamlit blue color for the chart
        st.bar_chart(prob_df, x='Class', y='Probability', color="#2980b9") 

    with col_data:
        st.dataframe(
            prob_df, 
            column_config={
                "Probability": st.column_config.ProgressColumn(
                    "Probability (%)",
                    format="%.2f",
                    min_value=0,
                    max_value=100,
                ),
            },
            hide_index=True,
        )
    
    status_text.empty()
    progress_bar.empty()

# --- Streamlit Main UI ---

st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload a brain MRI image (JPG, JPEG, PNG) to get a tumor prediction using a custom **Keras Deep Learning model**.")

st.markdown("---")

upload_col, display_col = st.columns([1, 1])
image = None

with upload_col:
    # Use st.container to make the uploader section stand out
    with st.container():
        uploaded_file = st.file_uploader(
            "📂 Choose an MRI Image...", 
            type=["jpg", "jpeg", "png"]
        )
        
        # Add a placeholder for spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # The button will inherit the custom CSS styling
        classify_button = st.button("🚀 Classify Image", disabled=uploaded_file is None)

with display_col:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(
            image, 
            caption='Uploaded MRI Image', 
            use_column_width=True,
            output_format="PNG"
        )
    elif uploaded_file is None:
        st.info("⬆️ Please upload an image to begin classification.")

# Handle classification trigger
if uploaded_file is not None and classify_button and image is not None:
    predict_and_display(image, MODEL)


st.markdown("---")
st.caption("""
    *Model Information:* This is a demonstration for a 4-class classification 
    (Glioma, Meningioma, No Tumor, Pituitary). Results are for informational 
    purposes only and **NOT** a substitute for professional medical advice.
""")