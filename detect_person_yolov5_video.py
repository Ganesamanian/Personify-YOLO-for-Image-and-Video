#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from PIL import Image
from IPython.display import display, Video
import torch


# In[2]:


# Load your YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')  # You need to replace this with your model loading code

def detect_person_yolov5_video(input_video_path, output_video_path, confidence_threshold=0.5):
    # Open the video file for reading
    cap = cv2.VideoCapture(input_video_path)
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the video's frame dimensions and frame rate
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))

    # Create VideoWriter object to save the annotated video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))
    frame_number = 0
    while True:
        ret, frame = cap.read()  # Read a frame from the video

        if not ret:
            break  # Break the loop when we reach the end of the video

        # Perform inference on the frame
        results = model(frame)

        # Get detected objects with 'person' class and their bounding boxes, class labels, and confidence scores
        detected_people = results.pred[0][results.pred[0][:, 5] == 0]  # Class index 0 corresponds to 'person'
        bounding_boxes = detected_people[:, :4].cpu().numpy()  # Extract bounding boxes
        confidences = detected_people[:, 4].cpu().numpy()  # Extract confidence scores

        # Get class names
        class_names = results.names[0]

        # Draw bounding boxes on the frame
        for i in range(len(bounding_boxes)):
            bbox = bounding_boxes[i]
            confidence = confidences[i]
            x_min, y_min, x_max, y_max = map(int, bbox)
            label = f"{class_names}: {confidence:.2f}"
            color = (0, 0, 255)  # Red color for bounding box
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (0, 0, 255)  # Red color for text
            line_type = cv2.LINE_AA

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

            # Add label (class name and confidence score) as text
            cv2.putText(frame, label, (x_min + 5, y_min + 15), font, font_scale, font_color, thickness, line_type)

        # Write the annotated frame to the output video
        out.write(frame)
        
        frame_number += 1
        progress_percentage = (frame_number / total_frames) * 100
        print(f"Processing Frame {frame_number}/{total_frames} ({progress_percentage:.2f}%)", end='\r')

    
    # Release video objects
    cap.release()
    out.release()


# In[3]:


input_video_path = 'Videos/Changing an Absorbent Brief for a Bed bound Patient.mp4'
output_video_path = 'Videos/output_Changing an Absorbent Brief for a Bed bound Patient.avi'
detect_person_yolov5_video(input_video_path, output_video_path)


# In[4]:


input_video_path = 'Videos/NHC - TERi™ Geriatric Care Trainer.mp4'
output_video_path = 'Videos/output_NHC - TERi™ Geriatric Care Trainer.avi'
detect_person_yolov5_video(input_video_path, output_video_path)


# In[5]:


input_video_path = 'Videos/A215.avi'
output_video_path = 'Videos/Output_A215.avi'
detect_person_yolov5_video(input_video_path, output_video_path)

