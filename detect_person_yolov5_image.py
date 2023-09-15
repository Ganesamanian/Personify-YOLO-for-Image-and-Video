#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import torch
import time
import numpy as np
from PIL import Image
from IPython.display import Image as IPImage, display


# In[2]:


# Load the YOLOv5 model (change 'yolov5s.pt' to the appropriate model checkpoint)
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)


# In[3]:


def detect_person_yolov5_r(image_path, confidence_threshold=0.5):
    img = Image.open(image_path)  # Open the image
    img = np.array(img)  # Convert to a NumPy array
    
    start_time = time.time()
    # Perform inference on the image
    results = model(img) 
    print(time.time()-start_time)

    # Get detected objects with 'person' class and their bounding box sizes
    detected_people = results.pred[0][results.pred[0][:, 5] == 0]  # Class index 0 corresponds to 'person'
    bounding_box_sizes = detected_people[:, 2:4].cpu().numpy()

#     # Display the image with bounding boxes
    results.show()

    # Print the bounding box sizes
    for i, (width, height) in enumerate(bounding_box_sizes, start=1):
        print(f"Person {i}: Width = {width:.2f}, Height = {height:.2f}")


detect_person_yolov5_r("Images/Person and dog.jpg")


# In[4]:


def detect_person_yolov5(image_path, confidence_threshold=0.5):
    img = Image.open(image_path)  # Open the image
    img = np.array(img)  # Convert to a NumPy array

    # Perform inference on the image
    results = model(img)

    # Get detected objects with 'person' class and their bounding boxes, class labels, and confidence scores
    detected_people = results.pred[0][results.pred[0][:, 5] == 0]  # Class index 0 corresponds to 'person'
    bounding_boxes = detected_people[:, :4].cpu().numpy()  # Extract bounding boxes
    confidences = detected_people[:, 4].cpu().numpy()  # Extract confidence scores

    # Get class names
    class_names = results.names[0]
    # Draw bounding boxes on the image
    for i in range(len(bounding_boxes)):
        bbox = bounding_boxes[i]
        confidence = confidences[i]
#         class_name = class_names[0]
        x_min, y_min, x_max, y_max = map(int, bbox)
        label = f"{class_names}: {confidence:.2f}"
        color = (0, 0, 255)  # Red color for bounding box
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 255)  # Red color for text
        line_type = cv2.LINE_AA

        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

        # Add label (class name and confidence score) as text
        cv2.putText(img, label, (x_min +5, y_min + 15), font, font_scale, font_color, thickness, line_type)


    
    # Display the annotated image using IPython.display
    display(IPImage(data=cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[1]))


# In[5]:


detect_person_yolov5("Images/Person and dog.jpg")


# In[6]:


detect_person_yolov5("Images/Person blanket.jpg")


# In[7]:


detect_person_yolov5("Images/Person blanket 2.jpg")


# In[8]:


detect_person_yolov5("Images/G170008_20230714103320353.bmp")


# In[9]:


detect_person_yolov5("Images/G170008_20230714104022091.bmp")


# In[10]:


detect_person_yolov5("Images/G170008_20230714104227139.bmp")


# In[11]:


detect_person_yolov5("Images/G170008_20230714104228957.bmp")


# In[12]:


detect_person_yolov5("Images/G170008_20230714104402709.bmp")


# In[13]:


detect_person_yolov5("Images/G170008_20230714104404532.bmp")


# In[14]:


detect_person_yolov5("Images/Old person.jpg")


# In[15]:


detect_person_yolov5("Images/Person on the bed.jpg")


# In[16]:


detect_person_yolov5("Images/Person side angle.jpg")


# In[17]:


detect_person_yolov5("Images/Lateral position.jpg")


# In[18]:


detect_person_yolov5("Images/Person fully covered.jpg")


# In[19]:


detect_person_yolov5("Images/Person with two Nurse.jpg")


# In[20]:


detect_person_yolov5("Images/Prone position.jpg")


# In[21]:


detect_person_yolov5("Images/Semi fowler position.jpg")


# In[22]:


detect_person_yolov5("Images/High Fowler position.jpg")


# In[23]:


detect_person_yolov5("Images/low fowler position.jpg")


# In[24]:


detect_person_yolov5("Images/Two Person.jpg")


# In[25]:


detect_person_yolov5("Images/Lateral position_foldedleg.jpg")


# In[26]:


detect_person_yolov5("Images/Covered.jpg")


# In[27]:


detect_person_yolov5("Images/Contorded_position.jpg")


# In[28]:


detect_person_yolov5("Images/Contorded_position2.jpg")


# In[29]:


detect_person_yolov5("Images/Contorded_position3.jpg")


# In[30]:


detect_person_yolov5("Images/Contorded_position4.jpg")


# In[31]:


detect_person_yolov5("Images/Old_person_blurr.jpg")


# In[32]:


detect_person_yolov5("Images/old_person_with pillow.jpg")


# In[33]:


detect_person_yolov5("Images/New1.bmp")


# In[34]:


detect_person_yolov5("Images/New2.bmp")


# In[35]:


detect_person_yolov5("Images/Wheelchair1.jpg")


# In[36]:


detect_person_yolov5("Images/Wheelchair2.bmp")


# In[37]:


detect_person_yolov5("Images/Wheelchair3.bmp")


# In[38]:


detect_person_yolov5("Images/Wheelchair4.bmp")

