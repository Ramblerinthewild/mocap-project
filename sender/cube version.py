import cv2
import mediapipe as mp
import socket
import json

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Socket setup
HOST, PORT = '127.0.0.1', 12347
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
print(f"[Sender] Waiting for Blender on {HOST}:{PORT}...")
conn, addr = sock.accept()
print(f"[Sender] Connected to Blender from {addr}")

cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            ring_tip = landmarks[16]
            position = {
                "ring_tip": {
                    "position": [ring_tip.x, ring_tip.y, ring_tip.z]
                }
            }
            conn.sendall((json.dumps(position) + "\n").encode('utf-8'))

        cv2.imshow("Ring Finger Tracker", cv2.flip(frame, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

finally:
    cap.release()
    conn.close()
    sock.close()
    cv2.destroyAllWindows()
