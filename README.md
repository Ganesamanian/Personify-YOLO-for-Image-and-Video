# Personify: Yolo for Image & Video

## Repository Purpose

Welcome to the Personify repository! This repository contains Python code designed for detecting and annotating "person" objects in both images and videos using YOLO (You Only Look Once) deep learning models. YOLO is a cutting-edge object detection system capable of real-time detection of multiple object classes in both images and videos. The primary use case for this repository is in Geriatric Care, where it plays a crucial role in person detection in both RGB and thermal images.

## Methods for Person Detection

### Method 1: YOLOv5 Person Detection in Videos

In this method, we leverage YOLOv5, a popular variant of YOLO, to detect "person" objects in videos. The code annotates the detected persons with bounding boxes and confidence scores. To use this method:

1. Ensure you have the necessary dependencies installed, as listed in the "Dependencies" section below.
2. Load your YOLOv5 model using your preferred method, and replace the model loading code in the script or use the existing model provided in the 'YOLOv5' folder.
3. Run the script `Stream_detector_v5.ipynb` from the 'YOLOv5' folder.
4. Specify the input video path and output video path as instructed in the file.

### Method 2: YOLOv3 Person Detection in Images

This method employs YOLOv3, another variant of YOLO, to detect "person" objects in images. The code annotates the detected persons with bounding boxes, class labels, confidence scores, and bounding box sizes. To use this method:

1. Ensure you have the required dependencies installed, as mentioned in the "Dependencies" section below.
2. Load your YOLOv3 model and class names using your preferred method, and replace the model loading code in the script or use the existing model provided in the 'YOLOv3' folder.
3. Open and run the `YOLOv3.ipynb` function with the image file path as an argument to perform person detection and display the annotated image.

To download the YOLOv3 weights file, you can include the following instructions in your README:

##### Download YOLOv3 Weights

Before using YOLOv3 for person detection in images, you need to download the YOLOv3 weights file. You can download the YOLOv3 weights from the official YOLO website or use a pre-trained weights file.

You can download the official YOLOv3 weights from the official YOLO website. Visit the following link to download the YOLOv3 weights file:

[YOLOv3 Weights (Official)](https://pjreddie.com/media/files/yolov3.weights)

After downloading the weights, place the `yolov3.weights` file in the same directory as the script or in a location accessible by the script.

### Method 3: YOLOv5, YOLOv7, and YOLOv8 Person Detection in Images

These methods are similar to Method 2 but can be found in their respective folders ('YOLOv5', 'YOLOv7', and 'YOLOv8').

### Method 4: YOLOv7 Person Detection in Videos

This method is similar to Method 1 but can be found in the 'YOLOv7' folder. Please ensure you have a GPU setup for faster processing.

## Version

This README corresponds to version 1.0 of the person detection code.

## Setup Instructions

### Summary of Setup

To set up and use the person detection code, follow these general steps:

1. Clone this repository to your local machine.
2. Install the required dependencies, as mentioned in the "Dependencies" section below.
3. Replace the model loading code in the scripts with your own YOLO model loading code.
4. Use the provided scripts/functions to perform person detection on images or videos.

### Configuration

No specific configuration is required for this repository. However, you may need to configure your deep learning framework and YOLO model according to your specific requirements.

### Dependencies

Ensure you have the following dependencies installed:

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Pillow (`PIL`)
- PyTorch
- Matplotlib (for YOLOv3 image detection only)

You can install these dependencies using pip:

```bash
pip install opencv-python numpy Pillow torch matplotlib
```

### Database Configuration

There is no need for database configuration in this repository.

## Running Tests

This repository does not include specific tests. Testing is generally performed by running the provided scripts/functions with your own input data.

## Deployment Instructions

Deployment instructions may vary depending on your use case. In general, you can deploy these scripts/functions on a server or edge device capable of running deep learning models and processing images/videos.

## Contribution Guidelines

If you wish to contribute to this repository, please follow these guidelines:

- Feel free to open issues to report bugs or suggest enhancements.
- If you want to contribute code changes, fork the repository, make your changes in a feature branch, and create a pull request.
- Code reviews will be conducted to ensure the quality of contributions.

### Writing Tests

If you add new features or functionality, consider adding corresponding tests to ensure they work as expected.

### Code Review

All code contributions will be reviewed for correctness, style, and adherence to best practices.

### Other Guidelines

Please adhere to the coding standards and best practices of the Python programming language when making contributions.

## Who Do I Talk To?

If you have questions or need assistance with this repository, you can contact:

- Repo Owner/Admin: Ganesamanian Kolappan
- mail-id: kganesamanianthanu@gmail.com

Feel free to reach out for any inquiries or collaboration opportunities related to this project.
