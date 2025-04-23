# import cv2
# import mediapipe as mp
# import json
# import numpy as np
# import time
# import socket
# from scipy.spatial.transform import Rotation
#
#
# # Socket setup
# HOST = '127.0.0.1'
# PORT = 12345
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect((HOST, PORT))
#
# # Initialize MediaPipe
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5)
#
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# time.sleep(1)
#
# def landmark_to_blender_coords(landmark):
#     return [landmark.x, landmark.y, -landmark.z]
#
# def midpoint(a, b):
#     return type(a)(x=(a.x + b.x)/2, y=(a.y + b.y)/2, z=(a.z + b.z)/2)
#
# def calculate_quaternion(parent, child, bone_type="default"):
#     parent_blender = landmark_to_blender_coords(parent)
#     child_blender = landmark_to_blender_coords(child)
#     direction = np.array(child_blender) - np.array(parent_blender)
#     direction /= np.linalg.norm(direction)
#
#
#     if bone_type == "left_arm":
#         rest_direction = np.array([1, 0, 0])
#     elif bone_type == "right_arm":
#         rest_direction = np.array([-1, 0, 0])
#     elif bone_type == "spine":
#         rest_direction = np.array([0, -1, 0])
#     elif bone_type == "leg":
#         rest_direction = np.array([0, 1, 0])
#     elif bone_type == "right_shoulder":
#         rest_direction = np.array([-1, 0, 0])
#     elif bone_type == "left_shoulder":
#         rest_direction = np.array([1, 0, 0])
#     elif bone_type == "neck":
#         rest_direction = np.array([0, -1, 0])
#     else:
#         rest_direction = np.array([0, 0, 1])
#     rot = Rotation.align_vectors([rest_direction], [direction])[0]
#     return rot.as_quat().tolist()  # [w, x, y, z]
#
# try:
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break
#
#         image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(image_rgb)
#
#         if results.pose_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
#                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
#             )
#
#         if results.pose_world_landmarks:
#             landmarks = results.pose_world_landmarks.landmark
#             mid_shoulders = midpoint(landmarks[11], landmarks[12])
#             mid_hips = midpoint(landmarks[23], landmarks[24])
#
#
#
#             data = {
#
#                 "mixamorig:LeftShoulder":{
#                     "rotation": calculate_quaternion(mid_shoulders, landmarks[11], "left_shoulder")
#             },
#                 "mixamorig:RightShoulder": {
#                     "rotation": calculate_quaternion(mid_shoulders, landmarks[12], "right_shoulder")
#                 },
#                 "mixamorig:LeftUpLeg": {
#                     "rotation": calculate_quaternion(landmarks[23], landmarks[25], "leg")
#                 },
#                 "mixamorig:RightUpLeg": {
#                     "rotation": calculate_quaternion(landmarks[24], landmarks[26], "leg")
#                 },
#                 "mixamorig:Spine": {
#                     "rotation": calculate_quaternion(mid_hips, mid_shoulders, "spine")
#                 },
#                 "mixamorig:LeftArm": {
#                     "rotation": calculate_quaternion(landmarks[11], landmarks[13], "left_arm")
#                 },
#                 "mixamorig:RightArm": {
#                     "rotation": calculate_quaternion(landmarks[12], landmarks[14], "right_arm")
#                 },
#                 "mixamorig:LeftForeArm": {
#                     "rotation": calculate_quaternion(landmarks[13], landmarks[15], "left_arm")
#                 },
#                 "mixamorig:RightForeArm": {
#                     "rotation": calculate_quaternion(landmarks[14], landmarks[16], "right_arm")
#                 },
#                 "mixamorig:LeftLeg": {
#                     "rotation": calculate_quaternion(landmarks[27], landmarks[25], "spine")
#                 },
#                 "mixamorig:RightLeg": {
#                     "rotation": calculate_quaternion(landmarks[28], landmarks[26], "spine")
#                 }
#             }
#
#             sock.sendall((json.dumps(data) + "\n").encode('utf-8'))
#
#         cv2.imshow('Pose Tracking', frame)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
# finally:
#     sock.close()
#     cap.release()
#     cv2.destroyAllWindows()
