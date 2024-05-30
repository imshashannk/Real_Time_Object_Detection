import cv2
import numpy as np
import datetime
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Initialize the main window
main_window = tk.Tk()
main_window.title("Motion Detection System")
main_window.geometry("1280x720")

# Initialize the recording variables
recording = False
video_writer = None
no_movement_frames = 0

# Initialize the capture object
cap = cv2.VideoCapture(0)

# Set the width and height of the frames
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set the codec and fps for video recording
codec = 'XVID'
fps = 20

# Initialize the background model
bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)

# Set the minimum contour area
min_contour_area = 1000

# Create the authentication window
auth_window = tk.Toplevel()
auth_window.title("Authentication")
auth_window.geometry("300x200")

# Create the username and password labels and entries
username_label = tk.Label(auth_window, text="Username:")
username_label.pack()
username_entry = tk.Entry(auth_window)
username_entry.pack()

password_label = tk.Label(auth_window, text="Password:")
password_label.pack()
password_entry = tk.Entry(auth_window, show="*")
password_entry.pack()

# Create the error label
error_label = tk.Label(auth_window, text="")
error_label.pack()

# Create the authenticate button
auth_button = tk.Button(auth_window, text="Authenticate", command=lambda: authenticate(username_entry, password_entry))
auth_button.pack()

# Create a function to update the frame
def update_frame():
    global bg_model, recording, video_writer, cap, width, height, min_contour_area, no_movement_frames

    # Read a frame from the capture object
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Update the background model
    fg_mask = bg_model.apply(gray)

    # Morphological opening to remove noise
    kernel = np.ones((3,3),np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Morphological closing to fill gaps
    kernel = np.ones((10,10),np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the foreground mask
    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    movement_detected = False

    # Loop over the contours
    for contour in contours:
        # Compute the area of the contour
        area = cv2.contourArea(contour)

        # If the area is greater than the minimum contour area, draw a bounding box around it
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            movement_detected = True

    # Add date and time to the frame
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
            if no_movement_frames > fps:  # Stop recording if no movement for more than one second
                recording = False

    # Display the frame
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, update_frame)

# Define the authenticate function
def authenticate(username_entry, password_entry):
    global auth_window, video_writer, width, height, codec, fps

    # Get the username and password from the entries
    username = username_entry.get()
    password = password_entry.get()

    # Check if the username and password are correct
    if username == "admin" and password == "password":
        # Hide the authentication window
        auth_window.withdraw()

        # Start the video writer
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "motion_" + timestamp + ".mp4"
        video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*codec), fps, (width, height))

        # Start updating the frames
        update_frame()
    else:
        # Show an error message
        error_label.config(text="Invalid username or password")

# Create a label to display the video
label = tk.Label(main_window)
label.pack()

# Start the authentication window
auth_window.mainloop()

# Release the capture object and video writer on program exit
cap.release()
if video_writer is not None:
    video_writer.release()
