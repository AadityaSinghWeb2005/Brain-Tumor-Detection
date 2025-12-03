import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Use Streamlit's caching mechanism to load the model only once.
# This significantly speeds up the application when users interact with widgets.
# We set allow_output_mutation=True because Keras models are mutable objects.
@st.cache_resource
def load_keras_model():
    """Loads the pre-trained Keras model."""
    try:
        model = load_model("brain_tumor_predictor.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Define the class names based on common datasets for 4-class brain tumor classification
# (e.g., glioma, meningioma, no tumor, pituitary). You should verify this order 
# from your training notebook's data preparation steps.
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
IMG_SIZE = (224, 224)
MODEL = load_keras_model()

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload a brain MRI image to get a tumor prediction using a custom Keras model.")

def preprocess_image(image):
    """Resizes and converts the image for model inference."""
    # Resize image to model's input shape (224x224)
    image = image.resize(IMG_SIZE)
    # Convert image to numpy array
    img_array = np.asarray(image)
    
    # Ensure image has 3 color channels (e.g., handles grayscale conversion)
    if img_array.ndim == 2:
        # Convert grayscale to RGB if only 2 dimensions (H, W)
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        # Drop alpha channel if present
        img_array = img_array[..., :3]

    # Model expects a batch dimension, so expand dimensions to (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # NOTE: The model internally has Rescaling and Normalization layers 
    # (Rescaling(scale=0.0039215...) and Normalization).
    # So we only need to provide the raw 224x224x3 float32 array. The model loading process 
    # ensures the right input data type is handled. We'll use np.float32 for consistency.
    return img_array.astype(np.float32)

uploaded_file = st.file_uploader(
    "Choose an MRI Image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    st.write("")

    # Button to trigger prediction
    if st.button("Classify Image"):
        with st.spinner('Model running inference...'):
            # 1. Preprocess the image
            processed_image = preprocess_image(image)
            
            # 2. Make the prediction
            predictions = MODEL.predict(processed_image)
            predicted_class_index = np.argmax(predictions)
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = predictions[0][predicted_class_index] * 100

            # 3. Display the results
            st.success(f"**Prediction Complete!**")
            
            st.markdown(
                f"The image is classified as **{predicted_class_name}** "
                f"with **{confidence:.2f}%** confidence."
            )
            
            # Show prediction breakdown (optional)
            st.markdown("---")
            st.subheader("Prediction Probabilities")
            
            # Create a dictionary for plotting
            prob_dict = {
                'Class': CLASS_NAMES,
                'Probability (%)': [p * 100 for p in predictions[0]]
            }
            
            st.bar_chart(prob_dict, x='Class', y='Probability (%)')

st.markdown("---")
st.markdown("This Streamlit application uses caching to prevent reloading the large model every time you interact with the interface. ")

