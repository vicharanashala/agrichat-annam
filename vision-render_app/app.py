import streamlit as st
import cv2
import tempfile
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os

# Load YOLO model
model = YOLO("best.pt")

# Class descriptions (all lowercase for case-insensitive matching)
class_descriptions = {
    "anthracnose": "Dark, sunken lesions on leaves and stems.",
    "healthy": "This leaf appears healthy with no visible signs of disease.",
    "powdery mildew": "White powdery spots on leaf surfaces.",
    "leaf blight": "Browning or yellowing of leaf margins and tips.",
    "corn rust leaf": "Orange-brown pustules on corn leaves, fungal infection."
}

# Streamlit page config
st.set_page_config(page_title="🌿 Vision_agrichat - Crop Disease Detection App", layout="wide")

# Header
st.markdown(    
    "<h2 style='text-align: center; color: #228B22;'>🌿 Vision_agrichat - Crop Disease Detection App</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>AI-based disease detection tool for healthier, smarter farming.</p>",
    unsafe_allow_html=True
)

# Sidebar - Upload Image
with st.sidebar:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a leaf or crop image", type=["JPG","JPEG","PNG"])

# Layout
left_col, right_col = st.columns(2)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    with left_col:
        st.image(image, caption="📷 Original Image", use_container_width=True)

    # Convert image to OpenCV BGR
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    with right_col:
        if st.button("🔍 Predict"):
            with st.spinner("Detecting crop diseases..."):
                # Resize image up slightly to prevent label cutoffs
                scale = 1.2
                resized_img = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale)

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                    cv2.imwrite(temp_file.name, resized_img)
                    temp_path = temp_file.name

                try:
                    results = model(temp_path, conf=0.25)
                finally:
                    os.remove(temp_path)

                # Plot result and resize back to fit
                img_with_boxes = results[0].plot()
                final_result = cv2.resize(img_with_boxes, (img_bgr.shape[1], img_bgr.shape[0]))
                img_rgb = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="✅ Detection Output", use_container_width=True)

                # Show detection summary
                st.markdown("---")
                st.subheader("📋 Detection Summary")

                if results[0].boxes and len(results[0].boxes.cls) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        class_name = model.names[cls_id]

                        st.write(f"**🧪 Class:** {class_name}")
                        st.write(f"**🔢 Confidence:** {conf * 100:.2f}%")

                        # Case-insensitive match
                        desc = class_descriptions.get(class_name.lower())
                        if desc:
                            st.info(f"📝 {desc}")
                        else:
                            st.warning("ℹ️ No description available.")
                        st.markdown("---")
                else:
                    st.warning("⚠️ No disease detected in the image.")

# Footer 
st.markdown(
    "<hr><p style='text-align: center; font-size: 13px;'>🚀 Smart Agriculture System | Powered by ANNAM.AI</p>",
    unsafe_allow_html=True
)
