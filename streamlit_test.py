import streamlit as st
import cv2
from src.anti_spoof import spoof_predict
import os
import time
from tensorflow.keras.models import load_model
st.set_page_config(page_title="Streamlit WebCam App")


st.cache_data()
def load_anti_spoof_model():
    model = load_model("./model/face_antispoofing_model.h5")
    return model

model = load_anti_spoof_model()


cap = cv2.VideoCapture(1)
frame_placeholder = st.empty()

# Directory to save frames
FRAME_DIR = "captured_frames"
if not os.path.exists(FRAME_DIR):
    os.makedirs(FRAME_DIR)

# Function to predict live/spoof based on three frames
def predict_spoof(file_paths):
    results = []
    for file in file_paths:
        prediction = spoof_predict(file, model)  # Replace this with your model prediction function
        results.append(prediction)  # Assuming binary classification: 0 (spoof) or 1 (live)
    return results

# Streamlit UI Layout
st.title("Live vs Spoof Detection")
st.write("Look into the camera when prompted to verify you're not a spoof.")

# Class to handle video processing from webcam
class WebTest():
    def __init__(self):
        self.frames = []
        self.frame_paths = []
        self.current_step = "center"
        self.instructions = {"center": "Look straight into the camera.",
                             "right": "Turn your head to the right.",
                             "left": "Turn your head to the left."}
        self.result_message = None
        self.frames_to_capture = 3

    def transform(self, frame):
        # img = frame.to_ndarray(format="bgr24")
        img = frame

        # Capture and save frames
        if len(self.frames) < self.frames_to_capture:
            st.info(self.instructions[self.current_step])
            self.frames.append(img)
            file_path = os.path.join(FRAME_DIR, f"{self.current_step}_{len(self.frames)}.jpg")
            # cv2.imwrite(file_path, img)
            cv2.imwrite(file_path,  cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self.frame_paths.append(file_path)

        # If enough frames are captured, run the model
        if len(self.frames) == self.frames_to_capture:
            predictions = predict_spoof(self.frame_paths)
            print(predictions)
            if all(pred < 0.5 for pred in predictions):  # Assuming threshold of 0.5 for live
                self.result_message = "Check! You're good to go. Follow the next instruction."
                if self.current_step == "center":
                    self.current_step = "right"
                elif self.current_step == "right":
                    self.current_step = "left"
                elif self.current_step == "left":
                    self.result_message = "Verification Complete. You're verified!"
            else:
                self.result_message = "Spoof detected! Please try again."

            # Reset frames and file paths for the next instruction
            self.frames = []
            self.frame_paths = []

        # Display message
        if self.result_message:
            st.success(self.result_message)
            self.result_message = None

        return img

# Start webcam stream and apply the video transformer


if st.button('Start verification'):
    start = time.time()
    web = WebTest()
    
    while cap.isOpened() and web.result_message!="Verification Complete. You're verified!":
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
        print(time.time() - start)
        if time.time() - start >= 5:
            web.transform(frame)
            start = time.time()
    
    # cap.release()
    # cv2.destroyAllWindows()
