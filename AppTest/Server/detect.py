
import torch
import cv2
import json
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./boxes.pt')
model.conf = 0.30 # Confidence threshold
model.max_det = 60 # Maximum detections, limited here by max 2048 bytes on server-client comms of detections (2048 / 34 bytes per detection)

print('Loaded YOLOv5 model.')
#model = torch.hub.load('path/to/yolov5', 'custom', path='path/to/best.pt', source='local')  # LOCAL REPO, USE WHEN NO INTERNET

def detect_raw(img_bytes):

    buff = np.fromstring(img_bytes, np.uint8)
    buff = buff.reshape(1, -1)
    img = cv2.imdecode(buff, cv2.IMREAD_COLOR)

    return model(img)

def detect(img_bytes):

    detections = detect_raw(img_bytes)
    detections_list = json.loads(detections.pandas().xyxy[0].to_json(orient="records")) # load detections from json

    out = b"DET" # Default bytes to avoid sending nothing with no detections

    for det in detections_list: # 34 bytes per detection
        det_bytes = b""

        #CLASS ID (upper limit 2^8)
        det_bytes += int(det['class']).to_bytes(1, 'little')

        #BBOX MIN (upper limit 2^64)
        det_bytes += int(det['xmin']).to_bytes(8, 'little')
        det_bytes += int(det['ymin']).to_bytes(8, 'little')

        # BBOX MAX (upper limit 2^64)
        det_bytes += int(det['xmax']).to_bytes(8, 'little')
        det_bytes += int(det['ymax']).to_bytes(8, 'little')

        #CONFIDENCE AS PERCENTAGE (ROUNDED) (upper limit 2^8)
        det_bytes += int(det['confidence']*100).to_bytes(1, 'little')

        out += det_bytes

    return out
