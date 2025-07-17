🌿 Vision_agrichat - Crop Disease Detection App

**Vision_agrichat** is an intelligent web application designed to detect crop diseases from leaf images using a **YOLOv12n** deep learning model. Built using **Streamlit** and deployed on **Render.com**, the app provides a fast and user-friendly interface for farmers, researchers, and agri-tech developers to identify diseases and take action promptly.

---

📌 Key Features

- 📤 Upload crop or leaf images directly through the web interface
- 🧠 Performs real-time disease detection using **YOLOv12n** (a lightweight object detection model)
- 📋 Displays predicted class, confidence score, and disease-specific descriptions
- 🖥️ Built with **Streamlit** – easily accessible through any modern web browser
- 🚀 One-click deployment on **Render.com** with automatic build and deployment

---

🧠 Model Info

- Model used: `YOLOv12n` (Ultralytics)
- Training dataset: Crop disease dataset (custom)
- Model file: `best.pt` (~18 MB)

---

📂 Project Structure

    vision_agrichat/
    ├── app.py             # Main Streamlit application
    ├── best.pt            # YOLOv12n trained weights
    ├── requirements.txt   # Python dependencies
    ├── render.yaml        # Render deployment configuration
    └── README.md          # Project documentation


---

🛠 Tech Stack
Python 
Streamlit 
Ultralytics YOLOv12n 
OpenCV & PIL for image processing 
Render.com for deployment 

---

🔍 Sample Prediction

Uploading a maize leaf with powdery mildew returns:
🧪 Class: Powdery Mildew
🔢 Confidence: 94.20%
📝 Description: White powdery spots on leaf surfaces.

---
