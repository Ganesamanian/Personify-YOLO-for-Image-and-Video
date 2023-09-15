# Personify: Yolo for Image&Video

## What is this repository for?

This repository contains Python code for detecting and annotating "person" objects in both images and videos using YOLO (You Only Look Once) deep learning models. YOLO is a state-of-the-art object detection system capable of real-time detection of multiple object classes in images and videos. The primary use case for this repository is in Geriatric Care, where it is utilized for person detection in both RGB and thermal images.

## Methods for Person Detection

### Method 1: YOLOv5 Person Detection in Videos

This method uses YOLOv5, a popular variant of YOLO, to detect "person" objects in videos and annotate them with bounding boxes and confidence scores. It includes a Python script `detect_person_yolov5_video.py` that takes an input video, performs person detection, and saves the annotated video with bounding boxes.

#### How to Use:
1. Make sure you have the required dependencies installed, as mentioned in the "Dependencies" section below.
2. Load your YOLOv5 model using your preferred method and replace the model loading code in the script.
3. Run the script with the following command:
   ```
   python detect_person_yolov5_video.py input_video_path output_video_path confidence_threshold
   ```
   - `input_video_path`: Path to the input video file.
   - `output_video_path`: Path to save the annotated video.
   - `confidence_threshold` (optional): Confidence threshold for object detection (default is 0.5).

### Method 2: YOLOv3 Person Detection in Images

This method uses YOLOv3, another variant of YOLO, to detect "person" objects in images and annotate them with bounding boxes, class labels, confidence scores, and bounding box sizes. It includes a Python function `detect_person_yolov3` that takes an image as input and displays the annotated image.

#### How to Use:
1. Make sure you have the required dependencies installed, as mentioned in the "Dependencies" section below.
2. Load your YOLOv3 model and class names using your preferred method and replace the model loading code in the script.
3. Call the `detect_person_yolov3` function with the image file path as an argument to perform person detection and display the annotated image.

To download the YOLOv3 weights file, you can include the following instructions in your README:

##### Download YOLOv3 Weights

Before using YOLOv3 for person detection in images, you need to download the YOLOv3 weights file. You can download the YOLOv3 weights from the official YOLO website or use a pre-trained weights file.

You can download the official YOLOv3 weights from the official YOLO website. Visit the following link to download the YOLOv3 weights file:

[YOLOv3 Weights (Official)](https://pjreddie.com/media/files/yolov3.weights)

After downloading the weights, place the `yolov3.weights` file in the same directory as the script or in a location accessible by the script.


### Method 3: YOLOv5 Person Detection in Images

This method uses YOLOv5 to detect "person" objects in images and annotate them with bounding boxes and confidence scores. It includes a Python script `detect_person_yolov5_image.py` that takes an input image, performs person detection, and displays the annotated image.

#### How to Use:
1. Make sure you have the required dependencies installed, as mentioned in the "Dependencies" section below.
2. Load your YOLOv5 model using your preferred method and replace the model loading code in the script.
3. Run the script with the following command:
   ```
   python detect_person_yolov5_image.py input_image_path confidence_threshold
   ```
   - `input_image_path`: Path to the input image file.
   - `confidence_threshold` (optional): Confidence threshold for object detection (default is 0.5).

## Version

This README corresponds to version 1.0 of the person detection code.

## How do I get set up?

### Summary of Set Up

To set up and use the person detection code, follow these general steps:

1. Clone this repository to your local machine.
2. Install the required dependencies, as mentioned in the "Dependencies" section below.
3. Replace the model loading code in the scripts with your own YOLO model loading code.
4. Use the provided scripts/functions to perform person detection on images or videos.

### Configuration

No specific configuration is required for this repository. However, you may need to configure your deep learning framework and YOLO model according to your needs.

### Dependencies

Ensure you have the following dependencies installed:

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Pillow (`PIL`)
- PyTorch
- Matplotlib (for YOLOv3 image detection only)

You can install these dependencies using pip:

```
pip install opencv-python numpy Pillow torch matplotlib
```

### Database Configuration

There is no database configuration required for this repository.

## How to Run Tests

This repository does not include specific tests. Testing is generally done by running the provided scripts/functions with your own input data.

## Deployment Instructions

Deployment instructions may vary depending on your use case. Generally, you can deploy these scripts/functions on a server or edge device capable of running deep learning models and processing images/videos.

## Contribution Guidelines

If you wish to contribute to this repository, please follow these guidelines:

- You can open issues to report bugs or suggest enhancements.
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

- Repo Owner/Admin: [Your Name]
- Other Community or Team Contact: [Optional]

Feel free to reach out for any inquiries or collaboration opportunities related to this project.