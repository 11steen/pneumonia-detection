import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your model
model = tf.keras.models.load_model(r"C:\Users\saksh\OneDrive\Desktop\code\code\Output_Backups\weights-014-0.1170.keras")

# Preprocess the image
def preprocess_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    image = cv2.resize(image, (224, 224))  # Resize to 224x224
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Title and description
st.title("Pneumonia Detection App")
st.write("Upload a chest X-ray image to predict the probability of pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("Processing image...")
    
    # Preprocess the image
    image = preprocess_image(uploaded_file)
    
    # Show the uploaded image
    st.image(image[0], caption="Uploaded X-ray", use_column_width=True, channels="RGB")
    
    # Make prediction
    prediction = model.predict(image)[0]
    pneumonia_prob = prediction[1] * 100  # Probability of pneumonia in percentage
    
    # Display result
    st.subheader("Prediction Results")
    st.write(f"Probability of Pneumonia: **{pneumonia_prob:.2f}%**")
    if pneumonia_prob > 50:
        st.error("The X-ray indicates a high likelihood of pneumonia. Please consult a doctor.")
    else:
        st.success("The X-ray indicates a low likelihood of pneumonia.")
    
    # Progress bar for probability
    st.progress(int(pneumonia_prob))
    
    # Graphical visualization
    fig, ax = plt.subplots()
    ax.bar(["Normal", "Pneumonia"], prediction, color=["green", "red"])
    ax.set_title("Prediction Probabilities")
    ax.set_ylabel("Probability")
    st.pyplot(fig)
    
    # Interactive feedback
    st.write("Was this result helpful?")
    feedback = st.radio("Please select:", ["Yes", "No"])
    if feedback == "Yes":
        st.write("Thank you for your feedback!")
    else:
        st.write("We're sorry to hear that. We’ll strive to improve.")

# Footer
st.write("---")
st.write("Developed by user | Powered by AI")
