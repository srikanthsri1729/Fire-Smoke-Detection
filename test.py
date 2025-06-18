import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
import torch

# Set page config FIRST
st.set_page_config(page_title="Fire & Smoke Detection", layout="wide")

# Cache model loading
@st.cache_resource
def load_model():
    model_path = r"C:\Users\P SRIKANTH\Downloads\fire-smoke.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.stop()
    return YOLO(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = load_model()

# Streamlit UI
st.title("🔥 Fire & Smoke Detection")
st.write("Upload an image to detect fire or smoke using YOLOv11.")

# Sidebar: Confidence and image size
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 0.8, 0.3, 0.01)
img_size = st.sidebar.selectbox("Input Image Size", [640, 960, 1280], index=0)
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Load and preprocess image
    try:
        image = Image.open(uploaded_file).convert("RGB")
        # Enhanced preprocessing
        image = ImageEnhance.Contrast(image).enhance(1.8)  # Stronger contrast
        image = ImageEnhance.Brightness(image).enhance(1.3)  # Stronger brightness
        # Histogram equalization for low-contrast images
        image_np = np.array(image)
        if debug_mode:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2YCrCb)
            image_np[:, :, 0] = cv2.equalizeHist(image_np[:, :, 0])
            image_np = cv2.cvtColor(image_np, cv2.COLOR_YCrCb2BGR)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_np)
        image = image.resize((img_size, img_size), Image.Resampling.LANCZOS)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        image_np = np.array(image)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    # Run detection
    with st.spinner("Running detection..."):
        results = model(image_np, conf=conf_threshold, iou=0.5, imgsz=img_size, augment=True)

    # Log raw output for debugging
    if debug_mode:
        st.write("Raw Model Output (Bounding Boxes, Confidence, Classes):")
        st.write(results[0].boxes.data.cpu().numpy())

    # Display detection details
    st.subheader("Detection Details")
    if results[0].boxes:
        data = []
        for box in results[0].boxes:
            cls = model.names[int(box.cls)]
            conf = float(box.conf)
            xyxy = box.xyxy.cpu().numpy()[0]
            data.append([cls, f"{conf:.2f}", xyxy])
            if conf > 0.7:
                st.error(f"🚨 {cls} detected with high confidence!")
        st.table({"Class": [d[0] for d in data], "Confidence": [d[1] for d in data], "Bounding Box": [d[2] for d in data]})
    else:
        st.warning("No fire or smoke detected.")
        # Save failing image for analysis
        fail_dir = "failed_detections"
        os.makedirs(fail_dir, exist_ok=True)
        fail_path = os.path.join(fail_dir, f"failed_{uploaded_file.name}")
        cv2.imwrite(fail_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        with open("no_detection_images.txt", "a") as f:
            f.write(f"Failed detection: {uploaded_file.name} (saved to {fail_path})\n")
        st.image(image_np, caption="Failed Detection Image", use_container_width=True)

    # Show result image
    result_img = results[0].plot(labels=True, conf=True)
    st.image(result_img, caption="Detection Result", use_container_width=True)

    # Download result
    st.download_button(
        label="Download Result Image",
        data=cv2.imencode(".jpg", result_img)[1].tobytes(),
        file_name="detection_result.jpg",
        mime="image/jpeg"
    )

# Display GPU status
st.sidebar.write(f"Using {'CUDA' if torch.cuda.is_available() else 'CPU'}")