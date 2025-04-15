# 🩺 Pneumonia Detection

This project develops an AI system using **Convolutional Neural Networks (CNNs)** to detect pneumonia from chest X-ray images. It classifies scans as **"Normal"** or **"Pneumonia"**, aiming to assist healthcare professionals in early and accurate diagnosis.

---

## 🧠 Key Features

- 📊 **Image Preprocessing**: Normalizes and resizes grayscale chest X-ray images for consistent model input.
- 🧪 **Model Training**: Uses CNNs (e.g., MobileNetV2) for accurate classification.
- 🎯 **Model Evaluation**: Evaluates performance using metrics like accuracy, precision, recall, and confusion matrix.
- 🖥️ **Visualization**: Displays training history and predictions using Matplotlib and Seaborn.
- 🌐 **Streamlit Interface**: Interactive web app for uploading X-rays and viewing AI predictions.

---

## 🧰 Tools & Libraries

- Python, TensorFlow/Keras
- NumPy, Pandas, Matplotlib, Seaborn
- OpenCV
- Streamlit

---

## 📁 Dataset

The chest X-ray dataset includes two classes:
- `Normal`: Healthy lung scans
- `Pneumonia`: Infected lung scans

> Dataset Source: [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## 🚀 How to Run

1. Clone the repository  
2. Install required dependencies  
3. Run `main.py` or `app.py` via Streamlit  
4. Upload chest X-ray image to test

```bash
streamlit run app.py
