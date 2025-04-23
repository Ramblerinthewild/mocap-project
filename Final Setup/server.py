from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Store video frames from different sources
frames = {}
lock = threading.Lock()

@app.route('/')
def index():
    """Serve the main page that allows web clients to send video"""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    with lock:
        if request.sid in frames:
            del frames[request.sid]
    print('Client disconnected:', request.sid)

@socketio.on('video-frame')
def handle_video_frame(data):
    """Handle incoming video frames from web clients"""
    try:
        # Extract the base64 image data (remove data URL prefix if present)
        image_data = data
        if ',' in data:
            image_data = data.split(',')[1]
            
        # Decode base64 image
        jpg_original = base64.b64decode(image_data)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
        
        # Process frame here (e.g., with mediapipe) if needed
        # processed_frame = process_with_mediapipe(frame)
        
        # Store the frame for this client
        with lock:
            frames[request.sid] = {
                'frame': frame,
                'timestamp': time.time(),
                'source': 'web'
            }
    except Exception as e:
        print(f"Error processing frame: {e}")

@app.route('/pi-video', methods=['POST'])
def handle_pi_video():
    """Handle incoming video frames from the Pi"""
    try:
        # Extract base64 image from request
        data = request.get_json()
        image_data = data.get('frame')
        
        # Decode base64 image
        jpg_original = base64.b64decode(image_data)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
        
        # Process frame here (e.g., with mediapipe) if needed
        # processed_frame = process_with_mediapipe(frame)
        
        # Store the frame with a special key for the Pi
        with lock:
            frames['pi_camera'] = {
                'frame': frame,
                'timestamp': time.time(),
                'source': 'pi'
            }
        
        return {'status': 'success'}
    except Exception as e:
        print(f"Error processing Pi frame: {e}")
        return {'status': 'error', 'message': str(e)}, 400

@app.route('/video-feed')
def video_feed():
    """Stream the consolidated video feed"""
    def generate():
        while True:
            # Create a combined view of all frames
            output_frame = create_combined_view()
            
            if output_frame is not None:
                # Encode the frame
                _, buffer = cv2.imencode('.jpg', output_frame)
                frame_bytes = buffer.tobytes()
                
                # Yield the frame in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30fps
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def create_combined_view():
    """Create a combined view of all video sources"""
    with lock:
        # Remove stale frames (older than 3 seconds)
        current_time = time.time()
        stale_keys = [k for k, v in frames.items() 
                      if current_time - v['timestamp'] > 3]
        for k in stale_keys:
            del frames[k]
            
        if not frames:
            return None
            
        # Determine layout based on number of sources
        num_sources = len(frames)
        
        if num_sources == 1:
            # Just return the single frame
            return list(frames.values())[0]['frame']
        
        # For multiple sources, create a grid layout
        grid_size = 320  # Each video will be 320x240
        cols = min(3, num_sources)
        rows = (num_sources + cols - 1) // cols
        
        # Create blank canvas
        output = np.zeros((rows * grid_size, cols * grid_size, 3), dtype=np.uint8)
        
        # Place each frame in the grid
        idx = 0
        for _, frame_data in frames.items():
            frame = frame_data['frame']
            if frame is None:
                continue
                
            # Resize frame to fit grid
            resized = cv2.resize(frame, (grid_size, grid_size))
            
            # Calculate position
            row = idx // cols
            col = idx % cols
            
            # Place in output
            output[row*grid_size:(row+1)*grid_size, 
                   col*grid_size:(col+1)*grid_size] = resized
            
            # Add label
            source = frame_data.get('source', 'unknown')
            cv2.putText(output, source, (col*grid_size + 10, row*grid_size + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            idx += 1
            
        return output

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    import os
    if not os.path.exists('templates'):
        os.makedirs('templates')
        
    # Create the HTML template
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Source Video System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; text-align: center; }
        #video-container { margin: 20px auto; max-width: 800px; }
        #my-video { width: 320px; height: 240px; background-color: #333; margin: 10px; }
        #server-feed { width: 100%; max-height: 600px; background-color: #333; margin: 10px; }
        .button { 
            background-color: #4CAF50; 
            border: none; 
            color: white; 
            padding: 10px 20px; 
            text-align: center; 
            text-decoration: none; 
            display: inline-block; 
            font-size: 16px; 
            margin: 4px 2px; 
            cursor: pointer; 
            border-radius: 4px;
        }
        .button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <h1>Multi-Source Video System</h1>
    
    <div id="video-container">
        <h2>My Camera</h2>
        <video id="my-video" autoplay muted></video>
        <div>
            <button id="start-button" class="button">Start Camera</button>
            <button id="stop-button" class="button" disabled>Stop Camera</button>
        </div>
        
        <h2>Combined Video Feed</h2>
        <img id="server-feed" src="/video-feed">
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script>
        const socket = io();
        let mediaStream = null;
        let isCapturing = false;
        const videoElem = document.getElementById('my-video');
        const startBtn = document.getElementById('start-button');
        const stopBtn = document.getElementById('stop-button');
        
        startBtn.addEventListener('click', startCamera);
        stopBtn.addEventListener('click', stopCamera);
        
        // Connect to Socket.IO server
        socket.on('connect', () => {
            console.log('Connected to server');
        });
        
        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            stopCamera();
        });
        
        async function startCamera() {
            try {
                // Request camera access
                mediaStream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480 },
                    audio: false 
                });
                
                // Display video in the element
                videoElem.srcObject = mediaStream;
                
                // Enable stop button, disable start button
                startBtn.disabled = true;
                stopBtn.disabled = false;
                
                // Start sending frames
                isCapturing = true;
                captureFrames();
                
            } catch (error) {
                console.error('Error accessing camera:', error);
                alert('Could not access camera. Please ensure you have granted camera permissions.');
            }
        }
        
        function stopCamera() {
            isCapturing = false;
            
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
            
            videoElem.srcObject = null;
            
            // Enable start button, disable stop button
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }
        
        function captureFrames() {
            if (!isCapturing) return;
            
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = videoElem.videoWidth;
            canvas.height = videoElem.videoHeight;
            
            // Capture frame and send
            context.drawImage(videoElem, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg', 0.7);
            socket.emit('video-frame', imageData);
            
            // Schedule next capture
            setTimeout(captureFrames, 100); // 10 fps to reduce bandwidth
        }
    </script>
</body>
</html>
        """)
    
    # Start the Flask-SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)