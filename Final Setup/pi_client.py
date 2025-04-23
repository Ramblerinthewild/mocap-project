from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import numpy as np
import mediapipe as mp
from threading import Lock

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Thread-safe frame storage
frames = {"pi5": None, "web_client": None}
lock = Lock()

@app.route('/')
def index():
    return render_template('index.html')  # Web client page

@socketio.on('pi5_stream')
def handle_pi5_frame(data):
    global frames
    jpg_original = base64.b64decode(data)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
    
    # Process with MediaPipe
    with lock:
        frames["pi5"] = frame
        process_and_display()

@socketio.on('web_client_stream')
def handle_web_client_frame(data):
    global frames
    jpg_original = base64.b64decode(data)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
    
    with lock:
        frames["web_client"] = frame
        process_and_display()

def process_and_display():
    """Process frames with MediaPipe and show results."""
    for source, frame in frames.items():
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, landmarks, mp_hands.HAND_CONNECTIONS
                    )
            
            cv2.imshow(source, frame)
            cv2.waitKey(1)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)