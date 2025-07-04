import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download

# --------------------------
# Helper Functions
# --------------------------

@st.cache_resource
def download_tflite_model(plant_type):
    model_filenames = {
        "Potato": "potato.tflite",
        "Tomato": "tomato.tflite",
        "Pepper": "pepper.tflite"
    }

    filename = model_filenames[plant_type]
    model_path = hf_hub_download(
        repo_id="mesakhbesta/plantdiseasefituksw",
        filename=filename,
        repo_type="model"
    )
    return model_path

def preprocess_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def load_tflite_model(plant_type):
    model_path = download_tflite_model(plant_type)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_tflite(interpreter, input_data, plant_type):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    if plant_type == "Pepper":
        prob = output_data[0][0]
        return np.array([1 - prob, prob])
    else:
        return output_data[0]

def get_labels(plant_type):
    if plant_type == "Potato":
        # ✅ URUTAN SESUAI MODEL:
        # Class 0 = Healthy, Class 1 = Late Blight, Class 2 = Early Blight
        return ["Early Blight","Late Blight","Healthy"]
    elif plant_type == "Tomato":
        return [
            "Leaf Mold",                 # 0
            "Yellow Leaf Curl Virus",   # 1
            "Bacterial Spot",           # 2
            "Septoria Leaf Spot",       # 3
            "Healthy",                  # 4
            "Spider Mites",             # 5
            "Early Blight",             # 6
            "Target Spot",              # 7
            "Late Blight",              # 8
            "Tomato Mosaic Virus"       # 9
        ]
    elif plant_type == "Pepper":
        return ["Healthy", "Bacterial Spot"]

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

st.markdown("""
<div style='text-align: center;'>
    <h1>🌿 Plant Disease Detector</h1>
    <p style='font-size: 17px;'>Check your plant's health by uploading or capturing a photo of its leaf</p>
</div>
<hr>
""", unsafe_allow_html=True)

st.sidebar.header("🔧 App Settings")
st.sidebar.markdown("""
🌱 **Welcome!**  
Detect leaf diseases in 🥔 Potato, 🍅 Tomato, and 🌶️ Pepper using AI.

**Steps:**  
1. Select plant type  
2. Upload or capture photo  
3. Get prediction!  
""")

plant_type = st.sidebar.selectbox("📌 Select Plant Type", ["Potato", "Tomato", "Pepper"])
input_mode = st.sidebar.radio("📷 Image Input", ["Upload Image", "Camera"])

# Image Input
image = None
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("📂 Upload a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
else:
    picture = st.camera_input("📸 Capture leaf image")
    if picture:
        image = Image.open(picture)

# Prediction
if image:
    st.image(image, caption="📍 Your Leaf Image", use_container_width=True)

    if st.button("🔍 Detect Disease"):
        try:
            interpreter = load_tflite_model(plant_type)
            input_image = preprocess_image(image)
            prediction = predict_tflite(interpreter, input_image, plant_type)

            labels = get_labels(plant_type)
            max_index = np.argmax(prediction)
            pred_label = labels[max_index]
            confidence = prediction[max_index]

            st.success(f"✅ **Prediction Result:** {pred_label}")
            st.markdown(f"🧠 *The model is {confidence:.2%} confident this leaf shows signs of* **{pred_label}**.")

            st.markdown("#### 📊 Class Probabilities:")
            for label, prob in zip(labels, prediction):
                st.write(f"- **{label}**: {prob:.2%}")
        except Exception as e:
            st.error(f"🚫 Error during prediction: {e}")
else:
    st.info("Please provide a leaf image to begin detection.")

# Footer
st.markdown("""
<hr>
<p style='text-align: center; font-size: 14px;'>Made with 💚 by <b>Glory Winner Team</b> | Plant Disease Detection App</p>
""", unsafe_allow_html=True)
