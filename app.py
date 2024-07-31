import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import sys
from pathlib import Path
import pathlib
temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath

@st.cache_resource
def load_model():
    print(Path.cwd())
    model = torch.hub.load('/workspaces/gdp-dashboard-1/yolov5', 'custom', path='/workspaces/gdp-dashboard-1/best.pt',source='local',force_reload=True)
    return model


# Function to perform prediction
def predict(image, model):
    # Convert the PIL image to a format that YOLOv5 expects
    img_array = np.array(image)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Perform prediction
    results = model([img_rgb])
    return results

# Function to draw bounding boxes on the image
def draw_boxes(image, results):
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

# Load model
model = load_model()

st.title("Dental Disease Detection")

uploaded_file = st.file_uploader("Upload a dental image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if model is not None:
        # Perform prediction
        results = predict(image, model)
        
        # Convert image for OpenCV processing
        img_with_boxes = np.array(image)
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes on the image
        img_with_boxes = draw_boxes(img_with_boxes, results)
        
        # Convert back to RGB for displaying with Streamlit
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        st.image(img_with_boxes, caption='Detected Diseases', use_column_width=True)
    else:
        st.error("Failed to load the model. Please check the model path.")
