# python
import asyncio
import websockets
import cv2
import base64
import numpy as np

async def receive_video():
    uri = "ws://<PI_IP_ADDRESS>:8765"  # Replace with Pi's IP
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv()
            jpg_original = base64.b64decode(data)
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
            cv2.imshow("Live Video", frame)

            if cv2.waitKey(1) == 27:  # ESC key to exit
                break

asyncio.run(receive_video())