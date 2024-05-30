# Real_Time_Object_Detection
A motion detection system using OpenCV, YOLO for object detection, and Haar Cascade for face detection. Features include video recording on movement detection, user authentication, and logging. Built with Tkinter for GUI. Requires Python, OpenCV, Tkinter, Pillow, and Numpy. Configure settings in the CONFIG dictionary.
Motion Detection System
This project implements a motion detection system using OpenCV, YOLO (You Only Look Once) object detection, and a Haar Cascade for face detection. The system captures video from a webcam, processes the frames to detect movement, faces, and objects, and records the video when movement is detected. The project is built with a graphical user interface (GUI) using Tkinter, and it includes an authentication window to restrict access.

Features
Motion Detection: Detects movement in the video frames using a background subtraction method.
Object Detection: Uses YOLO to detect and classify objects in the video frames.
Face Detection: Detects faces in the video frames using a Haar Cascade.
Recording: Records video when movement is detected and stops recording after a period of no movement.
Authentication: Includes a login window to authenticate users before accessing the system.
Logging: Logs significant events, such as authentication attempts and errors, to a file.
Requirements
Python 3.x
OpenCV
Tkinter
Pillow
Numpy
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/motion-detection-system.git
cd motion-detection-system
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Ensure you have the YOLO configuration and weight files (yolov4.cfg, yolov4.weights) and the class names file (coco.names). Place these files in the project directory.

Usage
Run the application:

bash
Copy code
python main.py
An authentication window will appear. Enter the username and password (default: admin/password).

After successful authentication, the main window will display the video feed from the webcam. The system will start processing frames to detect motion, objects, and faces.

When motion is detected, the system will start recording the video. The recording will stop after a period of no motion.

Configuration
The configuration settings for the application are stored in the CONFIG dictionary. Key settings include:

username and password: Credentials for authentication.
model_cfg, model_weights, model_classes: Paths to the YOLO configuration, weight, and class files.
min_confidence: Minimum confidence threshold for YOLO object detection.
nms_threshold: Non-maxima suppression threshold for YOLO object detection.
min_contour_area: Minimum contour area to consider for motion detection.
video_codec: Video codec for recording.
fps: Frames per second for recording.
Logging
The application logs events to a file named motion_detection.log. Logged events include successful and failed authentication attempts and errors encountered during execution.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
YOLO (You Only Look Once) - YOLO Website
OpenCV - OpenCV Website
Tkinter - Tkinter Documentation
Haar Cascade - OpenCV Haar Cascades
