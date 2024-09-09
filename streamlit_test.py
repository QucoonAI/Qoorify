import streamlit as st
import os
import cv2
import time
from datetime import datetime
import numpy as np


st.set_page_config(page_title="Streamlit WebCam App")
st.title("Qoorify Prototype")
st.caption("ID verification with OpenCV")
user_name=st.text_input("Enter your name")
if st.button('Start verification'):

    cap = cv2.VideoCapture(0)

    
    frame_placeholder = st.empty()

    
    frames = []

    start_time = time.time()
    start_time_formatted = datetime.fromtimestamp(start_time)
    start_time_str = start_time_formatted.strftime("%d-%m-%Y")
    
    def show_instruction_and_capture(instruction, delay=20, num_frames=3):
        st.write(instruction)
        start_time = time.time()
        
        
        while time.time() - start_time < delay:
            ret, frame = cap.read()
            if not ret:
                st.write("Video capture stopped or disconnected")
                return False
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")
        
        time.sleep(2)
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                st.write("Video capture stopped or disconnected")
                return False
                        
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_placeholder.image(frame, channels="RGB")
            
            
        
        return True

    
    if cap.isOpened():
        if not show_instruction_and_capture("Look straight at the camera", delay=4):
            st.write("Failed to capture frames.")
        
        if not show_instruction_and_capture("Look to the left", delay=4):
            st.write("Failed to capture frames.")
        
        if not show_instruction_and_capture("Look to the right", delay=4):
            st.write("Failed to capture frames.")

    
    cap.release()
    cv2.destroyAllWindows()
    
    user_folder_path = os.path.join(r"C:\Users\emman\Documents\python machine learning\streamlit\user_folder", user_name)

    if not os.path.exists(user_folder_path):
        os.makedirs(user_folder_path)
   
    
    for i, captured_frame in enumerate(frames):
        
            st.image(captured_frame, caption=f"Captured Frame {i+1}", channels="RGB")
            file_name = f"{user_name}_{i}__{start_time_str}.jpg"
            file_path = os.path.join(user_folder_path, file_name)
            success = cv2.imwrite(file_path,  cv2.cvtColor(captured_frame, cv2.COLOR_RGB2BGR))
        
            if success:
                st.write(f"Saved: {file_path}")
            else:
                st.write(f"Failed to save: {file_path}")
  
        

    

    
    
    
    
    

    
