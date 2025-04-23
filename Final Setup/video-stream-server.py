# python
import asyncio
import websockets
import cv2
import base64
import numpy as np
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

async def video_stream(websocket):
    while True:
        frame = picam2.capture_array()
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        await websocket.send(jpg_as_text)
        await asyncio.sleep(0.03)  # ~30fps

async def main():
    async with websockets.serve(video_stream, "0.0.0.0", 8765):
        await asyncio.Future()  # run forever

asyncio.run(main())