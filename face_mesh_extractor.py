import os
import cv2
import mediapipe as mp
import numpy as np

# Input directory
input_path = r'C:\Users\emman\Documents\python machine learning\Qoorify\good ones\Right'

# Output directory
output_path = r"C:\Users\emman\Documents\python machine learning\Qoorify\good ones\Right"


face_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
face_mesh=mp.solutions.face_mesh.FaceMesh()
marks= mp.solutions.drawing_utils



# Loop through the image files in the directory
for img_file in os.listdir(input_path):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):  # Ensure only image files are processed
        img_path = os.path.join(input_path, img_file)
        img1 = cv2.imread(img_path)
        
        if img1 is None:
            print(f"Error loading image {img_file}")
            continue
        
        # Convert the image to RGB because MediaPipe expects RGB input
        img_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        # Process the image with Face Mesh
        results = face_mesh.process(img_rgb)

        # Check if any face landmarks were detected
        if results.multi_face_landmarks:
            # Loop over the detected faces
            for face_landmarks in results.multi_face_landmarks:
                # Draw the landmarks on the image
               marks.draw_landmarks(img1,face_landmarks,face_connections,
                                                        marks.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                                        marks.DrawingSpec(color=(255, 0, 0), thickness=2)
                                                        )
        
        hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        # Save the image with the face mesh
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
    
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img1,img1, mask= mask)
        output_img_path = os.path.join(output_path, f"face_mesh_{img_file}")
        cv2.imwrite(output_img_path, res)
        print(f"Image saved to {output_img_path}")


