import cv2
import mediapipe as mp
import json
import numpy as np
import time
import socket
from scipy.spatial.transform import Rotation
from tkinter import Tk, Label, Button, Canvas
from threading import Thread, Lock
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageTk

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

# Global variables for 3D visualization
landmarks_3d = None
connections = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Spine
    (0, 4), (4, 5), (5, 6),          # Left arm
    (0, 8), (8, 9), (9, 10),         # Right arm
    (7, 11), (11, 13), (13, 15),     # Left leg
    (7, 12), (12, 14), (14, 16)      # Right leg
]

# Tkinter UI variables
root = None
video_label = None
connection_label = None
fig = None
ax = None
canvas = None


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
            if not blender_connected:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((HOST, PORT))
                    blender_connected = True
                    connection_label.config(text="Connected to Blender", fg="green")
                except Exception as e:
                    blender_connected = False
                    connection_label.config(text="Waiting for Blender connection...", fg="red")
        time.sleep(3)  # Retry every 3 seconds


def pose_estimator():
    """
    Processes the live video feed and updates the Tkinter video frame.
    """
    global cap, running, landmarks_3d

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

        # Send pose data to Blender
        if blender_connected and results.pose_world_landmarks:
            try:
                landmarks = results.pose_world_landmarks.landmark
                mid_shoulders = midpoint(landmarks[11], landmarks[12])
                mid_hips = midpoint(landmarks[23], landmarks[24])

                data = {
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

                sock.sendall((json.dumps(data) + "\n").encode('utf-8'))
            except Exception as e:
                print(f"Error sending data to Blender: {e}")


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


def start_pose_estimation():
    global running
    running = True
    Thread(target=pose_estimator, daemon=True).start()
    Thread(target=update_3d_pose, daemon=True).start()
    Thread(target=manage_blender_connection, daemon=True).start()


def stop_pose_estimation():
    global running, cap, sock
    running = False
    if cap:
        cap.release()
    if sock:
        sock.close()
    video_label.config(image="")
    connection_label.config(text="Stopped", fg="black")


# Tkinter UI
root = Tk()
root.title("Pose Estimation Interface")

# Video feed
video_label = Label(root)
video_label.pack(side="left")

# 3D pose visualization
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side="right")

# Connection status
connection_label = Label(root, text="Waiting for Blender connection...", fg="red")
connection_label.pack()

# Start/Stop buttons
start_button = Button(root, text="Start Pose Estimation", command=start_pose_estimation)
start_button.pack()
stop_button = Button(root, text="Stop Pose Estimation", command=stop_pose_estimation)
stop_button.pack()

root.mainloop()