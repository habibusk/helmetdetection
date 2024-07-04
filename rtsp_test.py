#import semua library yang dibutuhkan
import os
import io
import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import streamlit as st
import torch
import time
import pandas as pd

# Setting page layout
st.set_page_config(
    page_title="YoloS Helmet Detection",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded"
)

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(bytes_data))  # Convert bytes data to PIL image
        return image
    else:
        raise FileNotFoundError("No file uploaded")

# Function to convert OpenCV image to PIL image
def cv2_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def draw_bounding_boxes(image, results, model, confidence):
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score.item() >= confidence:
            box = [int(i) for i in box.tolist()]
            draw.rectangle(box, outline="purple", width=2)
            label_text = f"{model.config.id2label[label.item()]} ({round(score.item(), 2)})"
            draw.text((box[0], box[1]), label_text, fill="white")
    return image

def process_image(image, model, processor, confidence):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=confidence, target_sizes=target_sizes)[0]
    return results

def detection_results_to_dict(results, model):
    detection_dict = {
        "objects": []
    }
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        detection_dict["objects"].append({
            "label": model.config.id2label[label.item()],
            "confidence": round(score.item(), 3),
            "box": box
        })
    return detection_dict

def convert_dict_to_csv(detection_dict_list):
    combined_results = []
    for detection_dict in detection_dict_list:
        combined_results.extend(detection_dict["objects"])
    df = pd.DataFrame(combined_results)
    return df.to_csv(index=False).encode('utf-8')

def clear_detection_results():
    st.session_state.detection_dict_list = []

# Initialize session state to store detection results
if 'detection_dict_list' not in st.session_state:
    st.session_state.detection_dict_list = []

# Streamlit App Configuration
st.header("Helmet Rider Detection")

# Sidebar for Model Selection and Confidence Slider
st.sidebar.header("ML Model Config")
models = ["./weights/gghsgn/final200" ,"./weights/gghsgn/final100", "./weights/gghsgn/final50"]

model_name = st.sidebar.selectbox("Select model", models)
confidence = st.sidebar.slider("Select Model Confidence", 25, 100, 40, step=5) / 100

# Load Model and Processor
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForObjectDetection.from_pretrained(model_name)

# Option to select Real-Time or Upload Image
mode = st.sidebar.selectbox("Select Input Mode", ["Upload Image", "Real-Time Webcam", "RTSP Video"])

# Option if select Upload Image
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
    else:
        image = None
    submit = st.button("Detect Objects")
    if submit and image is not None:
        try:
            image_data = input_image_setup(uploaded_file)
            st.subheader("The response is..")
            
            results = process_image(image, model, processor, confidence)
            drawn_image = draw_bounding_boxes(image.copy(), results, model, confidence)
            st.image(drawn_image, caption="Detected Objects", use_column_width=True)
            
            st.subheader("List of Objects:")
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                st.write(
                    f"Detected :orange[{model.config.id2label[label.item()]}] with confidence "
                    f":green[{round(score.item(), 3)}] at location :violet[{box}]"
                )

            detected_objects = {model.config.id2label[label.item()]: 0 for label in results["labels"]}
            for label in results["labels"]:
                detected_objects[model.config.id2label[label.item()]] += 1
            for obj, count in detected_objects.items():
                st.write(f"Class :orange[{obj}] detected {count} time(s)")

            detection_dict = detection_results_to_dict(results, model)
            #st.write(detection_dict)

            csv_data = convert_dict_to_csv([detection_dict])
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name="detection_results.csv",
                mime="text/csv"
            )
        
        except Exception as e:
            st.error(f"Error: {e}")

    elif submit and image is None:
        st.error("Please upload an image before trying to detect objects.")

# Option if select Realtime Webcam
elif mode == "Real-Time Webcam":
    run = st.checkbox("Run Webcam")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam")
                break
            
            frame_pil = cv2_to_pil(frame)
            
            try:
                results = process_image(frame_pil, model, processor, confidence)
                drawn_image = draw_bounding_boxes(frame_pil.copy(), results, model, confidence)
                FRAME_WINDOW.image(drawn_image, caption="Detected Objects", use_column_width=True)
                
                st.subheader("List of Objects:")
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = [round(i, 2) for i in box.tolist()]
                    st.write(
                        f"Detected :orange[{model.config.id2label[label.item()]}] with confidence "
                        f":green[{round(score.item(), 3)}] at location :violet[{box}]"
                    )

                detected_objects = {model.config.id2label[label.item()]: 0 for label in results["labels"]}
                for label in results["labels"]:
                    detected_objects[model.config.id2label[label.item()]] += 1
                for obj, count in detected_objects.items():
                    st.write(f"Class :orange[{obj}] detected {count} time(s)")

                detection_dict = detection_results_to_dict(results, model)
                st.session_state.detection_dict_list.append(detection_dict)
                #st.write(detection_dict)
                
            
            except Exception as e:
                st.error(f"Error: {e}")

            time.sleep(0.1)  # Delay for the next frame capture to create an illusion of real-time
        cap.release()

    if not run and st.session_state.detection_dict_list:
        st.write("Detection stopped.")
        csv_data = convert_dict_to_csv(st.session_state.detection_dict_list)
        st.download_button(
            label="Download All Results as CSV",
            data=csv_data,
            file_name="all_detection_results.csv",
            mime="text/csv"
        )
        st.button("Clear Results", on_click=clear_detection_results)

# Option if select Realtime RTSP Video
elif mode == "RTSP Video":
    rtsp_url = st.text_input("RTSP URL")
    run = st.checkbox("Run RTSP Video")
    FRAME_WINDOW = st.image([])

    if rtsp_url and run:
        cap = cv2.VideoCapture(rtsp_url)
        st.subheader("List of Objects:")
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from RTSP stream")
                break
            
            frame_pil = cv2_to_pil(frame)
            
            try:
                results = process_image(frame_pil, model, processor, confidence)
                drawn_image = draw_bounding_boxes(frame_pil.copy(), results, model, confidence)
                FRAME_WINDOW.image(drawn_image, caption="Detected Objects", use_column_width=True)
                st.subheader("List of Objects:")
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = [round(i, 2) for i in box.tolist()]
                    st.write(
                        f"Detected :orange[{model.config.id2label[label.item()]}] with confidence "
                        f":green[{round(score.item(), 3)}] at location :violet[{box}]"
                    )

                detected_objects = {model.config.id2label[label.item()]: 0 for label in results["labels"]}
                for label in results["labels"]:
                    detected_objects[model.config.id2label[label.item()]] += 1
                for obj, count in detected_objects.items():
                    st.write(f"Class :orange[{obj}] detected {count} time(s)")

                detection_dict = detection_results_to_dict(results, model)
                st.session_state.detection_dict_list.append(detection_dict)
                #st.write(detection_dict)
            
            except Exception as e:
                st.error(f"Error: {e}")

            time.sleep(0.1)  # Delay for the next frame capture to create an illusion of real-time

        cap.release()

    if not run and st.session_state.detection_dict_list:
        st.write("Detection stopped.")
        csv_data = convert_dict_to_csv(st.session_state.detection_dict_list)
        st.download_button(
            label="Download All Results as CSV",
            data=csv_data,
            file_name="all_detection_results.csv",
            mime="text/csv"
        )
        st.button("Clear Results", on_click=clear_detection_results)
    elif run:
        st.error("Please provide a valid RTSP URL before running the stream.")

    # Ensure the video capture object is released if the checkbox is unchecked
    if 'cap' in locals():
        cap.release()
