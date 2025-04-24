import cv2
import mediapipe as mp
import json
import numpy as np
import time
import socket
from scipy.spatial.transform import Rotation
from tkinter import Tk, Label, Button, Canvas, Frame, StringVar, Radiobutton, filedialog
from threading import Thread, Lock
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageTk
import os

# Socket setup
HOST = '127.0.0.1'
PORT = 12345
sock = None
blender_connected = False
connection_lock = Lock()  # Lock for managing connection attempts

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5)

# Camera variables
cap = None
running = False

# Recording variables
recording = False
record_file = None
recorded_data = []

# Global variables for 3D visualization
landmarks_3d = None
connections = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Spine
    (0, 4), (4, 5), (5, 6),  # Left arm
    (0, 8), (8, 9), (9, 10),  # Right arm
    (7, 11), (11, 13), (13, 15),  # Left leg
    (7, 12), (12, 14), (14, 16)  # Right leg
]

# Tkinter UI variables
root = None
video_label = None
connection_label = None
fig = None
ax = None
canvas = None
mode_var = None
file_path_label = None
file_path = ""


def landmark_to_blender_coords(landmark):
    return [landmark.x, landmark.y, -landmark.z]


def midpoint(a, b):
    return type(a)(x=(a.x + b.x) / 2, y=(a.y + b.y) / 2, z=(a.z + b.z) / 2)


def calculate_quaternion(parent, child, bone_type="default"):
    parent_blender = landmark_to_blender_coords(parent)
    child_blender = landmark_to_blender_coords(child)
    direction = np.array(child_blender) - np.array(parent_blender)
    direction /= np.linalg.norm(direction)

    rest_direction = {
        "left_arm": np.array([1, 0, 0]),
        "right_arm": np.array([-1, 0, 0]),
        "spine": np.array([0, -1, 0]),
        "leg": np.array([0, 1, 0]),
        "right_shoulder": np.array([-1, 0, 0]),
        "left_shoulder": np.array([1, 0, 0]),
        "neck": np.array([0, -1, 0])
    }.get(bone_type, np.array([0, 0, 1]))

    rot = Rotation.align_vectors([rest_direction], [direction])[0]
    return rot.as_quat().tolist()


def manage_blender_connection():
    """
    Continuously manage the connection to Blender in a separate thread.
    """
    global blender_connected, sock

    while running:
        with connection_lock:
            if not blender_connected and mode_var.get() == "realtime":
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((HOST, PORT))
                    blender_connected = True
                    connection_label.config(text="Connected to Blender", fg="green")
                except Exception as e:
                    blender_connected = False
                    connection_label.config(text="Waiting for Blender connection...", fg="red")
        time.sleep(3)  # Retry every 3 seconds


def create_pose_data(landmarks):
    """
    Create pose data dictionary from landmarks
    """
    mid_shoulders = midpoint(landmarks[11], landmarks[12])
    mid_hips = midpoint(landmarks[23], landmarks[24])

    return {
        "timestamp": time.time(),
        "bones": {
            "mixamorig:LeftShoulder": {
                "rotation": calculate_quaternion(mid_shoulders, landmarks[11], "left_shoulder")
            },
            "mixamorig:RightShoulder": {
                "rotation": calculate_quaternion(mid_shoulders, landmarks[12], "right_shoulder")
            },
            "mixamorig:LeftUpLeg": {
                "rotation": calculate_quaternion(landmarks[23], landmarks[25], "leg")
            },
            "mixamorig:RightUpLeg": {
                "rotation": calculate_quaternion(landmarks[24], landmarks[26], "leg")
            },
            "mixamorig:Spine": {
                "rotation": calculate_quaternion(mid_hips, mid_shoulders, "spine")
            },
            "mixamorig:LeftArm": {
                "rotation": calculate_quaternion(landmarks[11], landmarks[13], "left_arm")
            },
            "mixamorig:RightArm": {
                "rotation": calculate_quaternion(landmarks[12], landmarks[14], "right_arm")
            },
            "mixamorig:LeftForeArm": {
                "rotation": calculate_quaternion(landmarks[13], landmarks[15], "left_arm")
            },
            "mixamorig:RightForeArm": {
                "rotation": calculate_quaternion(landmarks[14], landmarks[16], "right_arm")
            },
            "mixamorig:LeftLeg": {
                "rotation": calculate_quaternion(landmarks[27], landmarks[25], "spine")
            },
            "mixamorig:RightLeg": {
                "rotation": calculate_quaternion(landmarks[28], landmarks[26], "spine")
            }
        }
    }


def pose_estimator():
    """
    Processes the live video feed and updates the Tkinter video frame.
    """
    global cap, running, landmarks_3d, recording, recorded_data

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(1)

    while running:
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Draw landmarks on live video feed
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        # Extract 3D landmarks for stick figure
        if results.pose_world_landmarks:
            landmarks_3d = [
                (lm.x, -lm.z, -lm.y) for lm in results.pose_world_landmarks.landmark
            ]

        # Update the live video in Tkinter
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        video_label.config(image=image)
        video_label.image = image

        # Handle pose data based on mode
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            pose_data = create_pose_data(landmarks)

            if mode_var.get() == "realtime" and blender_connected:
                try:
                    # Send data in real-time to Blender
                    sock.sendall((json.dumps(pose_data["bones"]) + "\n").encode('utf-8'))
                except Exception as e:
                    print(f"Error sending data to Blender: {e}")

            elif mode_var.get() == "record" and recording:
                # Save data for recording
                recorded_data.append(pose_data)

                # Update status with frame count
                connection_label.config(text=f"Recording: {len(recorded_data)} frames", fg="red")


def update_3d_pose():
    """
    Updates the 3D stick figure visualization with landmarks and connections.
    """
    global running, landmarks_3d

    while running:
        if landmarks_3d:
            ax.clear()
            ax.set_title("3D Pose Estimation")
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

            # Plot the transformed landmarks
            x_vals = [lm[0] for lm in landmarks_3d]
            y_vals = [lm[1] for lm in landmarks_3d]
            z_vals = [lm[2] for lm in landmarks_3d]
            ax.scatter(x_vals, y_vals, z_vals, c='r', marker='o')

            for connection in connections:
                start, end = connection
                ax.plot(
                    [x_vals[start], x_vals[end]],
                    [y_vals[start], y_vals[end]],
                    [z_vals[start], z_vals[end]],
                    c='b'
                )

            canvas.draw()
        time.sleep(0.1)  # Small delay to prevent consuming too much CPU


def start_pose_estimation():
    global running, recording, recorded_data

    # Check if in record mode and file path is selected
    if mode_var.get() == "record" and not file_path:
        choose_file_path()
        if not file_path:  # User canceled
            return

    running = True

    if mode_var.get() == "record":
        recording = True
        recorded_data = []  # Clear previous data
        connection_label.config(text="Recording started...", fg="red")
    else:
        connection_label.config(text="Waiting for Blender connection...", fg="red")

    # Disable mode selection during operation
    for radio in mode_radios:
        radio.config(state="disabled")
    choose_file_button.config(state="disabled")

    Thread(target=pose_estimator, daemon=True).start()
    Thread(target=update_3d_pose, daemon=True).start()

    if mode_var.get() == "realtime":
        Thread(target=manage_blender_connection, daemon=True).start()


def stop_pose_estimation():
    global running, cap, sock, recording

    running = False

    if cap:
        cap.release()

    if sock:
        sock.close()

    # Save recorded data if in record mode
    if mode_var.get() == "record" and recording:
        save_recorded_data()
        recording = False

    video_label.config(image="")
    connection_label.config(text="Stopped", fg="black")

    # Re-enable mode selection
    for radio in mode_radios:
        radio.config(state="normal")
    choose_file_button.config(state="normal")


def save_recorded_data():
    """
    Save recorded pose data to the selected JSON file
    """
    if not recorded_data:
        connection_label.config(text="No data to save", fg="orange")
        return

    try:
        with open(file_path, 'w') as f:
            json.dump({"frames": recorded_data}, f, indent=2)
        connection_label.config(text=f"Recording saved: {len(recorded_data)} frames", fg="green")
    except Exception as e:
        connection_label.config(text=f"Error saving: {str(e)}", fg="red")


def choose_file_path():
    """
    Open file dialog to choose where to save the recorded data
    """
    global file_path

    # Make sure the file has .json extension
    filepath = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")],
        title="Save Recording As"
    )

    if filepath:
        file_path = filepath
        file_path_label.config(text=f"Save to: {os.path.basename(file_path)}")
    return filepath


def toggle_file_path_controls():
    """
    Show/hide file path controls based on selected mode
    """
    if mode_var.get() == "record":
        file_path_frame.pack(pady=5)
    else:
        file_path_frame.pack_forget()


# Tkinter UI
root = Tk()
root.title("Pose Estimation Interface")

# Mode selection frame
mode_frame = Frame(root)
mode_frame.pack(pady=10)

mode_var = StringVar(value="realtime")
Label(mode_frame, text="Mode:").pack(side="left")
mode_radios = []

mode_radios.append(Radiobutton(mode_frame, text="Real-time", variable=mode_var,
                               value="realtime", command=toggle_file_path_controls))
mode_radios[0].pack(side="left", padx=10)

mode_radios.append(Radiobutton(mode_frame, text="Record", variable=mode_var,
                               value="record", command=toggle_file_path_controls))
mode_radios[1].pack(side="left", padx=10)

# File path selection frame (initially hidden)
file_path_frame = Frame(root)
Label(file_path_frame, text="Recording File:").pack(side="left")
file_path_label = Label(file_path_frame, text="No file selected", width=30)
file_path_label.pack(side="left", padx=5)
choose_file_button = Button(file_path_frame, text="Choose File", command=choose_file_path)
choose_file_button.pack(side="left")

# Main content frame
content_frame = Frame(root)
content_frame.pack(fill="both", expand=True)

# Video feed
video_label = Label(content_frame)
video_label.pack(side="left")

# 3D pose visualization
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=content_frame)
canvas.get_tk_widget().pack(side="right")

# Bottom controls frame
controls_frame = Frame(root)
controls_frame.pack(pady=10)

# Connection status
connection_label = Label(controls_frame, text="Ready", fg="black")
connection_label.pack()

# Start/Stop buttons
button_frame = Frame(controls_frame)
button_frame.pack(pady=5)
start_button = Button(button_frame, text="Start", command=start_pose_estimation, width=10)
start_button.pack(side="left", padx=5)
stop_button = Button(button_frame, text="Stop", command=stop_pose_estimation, width=10)
stop_button.pack(side="left", padx=5)

# Initial UI setup based on mode
toggle_file_path_controls()

root.mainloop()