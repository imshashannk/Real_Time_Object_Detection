import cv2
import numpy as np
import datetime
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import logging

# Set up logging
logging.basicConfig(filename="motion_detection.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
CONFIG = {
    "username": "admin",
    "password": "password",
    "model_cfg": "yolov4.cfg",  # Ensure this path is correct
    "model_weights": "yolov4.weights",  # Ensure this path is correct
    "model_classes": "coco.names",  # Ensure this path is correct
    "min_confidence": 0.5,
    "nms_threshold": 0.4,
    "min_contour_area": 1000,
    "video_codec": "mp4v",
    "fps": 20,
}


# Load YOLO
def load_yolo(cfg, weights, classes_file):
    try:
        net = cv2.dnn.readNet(weights, cfg)
        with open(classes_file, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        return net, classes, output_layers
    except FileNotFoundError as e:
        messagebox.showerror("Error", f"File not found: {e}")
        logging.error(f"File not found: {e}")
        exit()
    except Exception as e:
        messagebox.showerror("Error", f"Error loading YOLO: {e}")
        logging.error(f"Error loading YOLO: {e}")
        exit()


# Load YOLO and Haar Cascade
net, classes, output_layers = load_yolo(CONFIG["model_cfg"], CONFIG["model_weights"], CONFIG["model_classes"])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the main window
main_window = tk.Tk()
main_window.title("Motion Detection System")
main_window.geometry("1280x720")

# Initialize the recording variables
recording = False
video_writer = None
no_movement_frames = 0

# Initialize the capture object
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise Exception("Could not open video device")

# Set the width and height of the frames
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the background model
bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)

# Create the authentication window
auth_window = tk.Toplevel()
auth_window.title("Authentication")
auth_window.geometry("300x200")

# Create the username and password labels and entries
username_label = ttk.Label(auth_window, text="Username:")
username_label.grid(row=0, column=0, padx=10, pady=10, sticky="W")
username_entry = ttk.Entry(auth_window)
username_entry.grid(row=0, column=1, padx=10, pady=10, sticky="E")

password_label = ttk.Label(auth_window, text="Password:")
password_label.grid(row=1, column=0, padx=10, pady=10, sticky="W")
password_entry = ttk.Entry(auth_window, show="*")
password_entry.grid(row=1, column=1, padx=10, pady=10, sticky="E")

# Create the error label
error_label = ttk.Label(auth_window, text="")
error_label.grid(row=2, columnspan=2, padx=10, pady=10)

# Create the authenticate button
auth_button = ttk.Button(auth_window, text="Authenticate", command=lambda: authenticate(username_entry.get(), password_entry.get()))
auth_button.grid(row=3, columnspan=2, pady=10)

# Create a label to display the video
video_label = ttk.Label(main_window)
video_label.pack()


def process_frame():
    global recording, video_writer, no_movement_frames

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_model.apply(gray)

        kernel_open = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        kernel_close = np.ones((10, 10), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        movement_detected = False

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > CONFIG["min_contour_area"]:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                movement_detected = True

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONFIG["min_confidence"]:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIG["min_confidence"], CONFIG["nms_threshold"])
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if movement_detected:
            if not recording:
                recording = True
                no_movement_frames = 0
            video_writer.write(frame)
            no_movement_frames = 0
        else:
            if recording:
                no_movement_frames += 1
                if no_movement_frames > CONFIG["fps"]:
                    recording = False

        cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cvimage)
        imgtk = ImageTk.PhotoImage(image=img)

        # Updating video label in a thread-safe manner
        video_label.after(0, lambda: update_video_label(imgtk))


def update_video_label(imgtk):
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)


def authenticate(username, password):
    global auth_window, video_writer, width, height

    if username == CONFIG["username"] and password == CONFIG["password"]:
        auth_window.withdraw()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motion_{timestamp}.mp4"
        video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*CONFIG["video_codec"]), CONFIG["fps"],
                                       (width, height))
        threading.Thread(target=process_frame, daemon=True).start()
        logging.info("Authentication successful")
    else:
        error_label.config(text="Invalid username or password")
        logging.warning("Authentication failed")


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        cap.release()
        if video_writer is not None:
            video_writer.release()
        main_window.destroy()


main_window.protocol("WM_DELETE_WINDOW", on_closing)

auth_window.mainloop()
