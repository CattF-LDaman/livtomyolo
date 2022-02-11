
from . import HNDSHK
from .dataset import class_mapping

import socket
import threading
import time
import io
import sys
import cv2
import numpy
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

class Fetcher:

    def __init__(self, server_address, server_port, camera=0):

        self.conn = socket.socket()
        self.conn.connect((server_address, server_port))
        print(f'Connected to {server_address}:{server_port}')

        self.conn.send(HNDSHK)

        self.wait_time = 0

        self.fetching = True
        self.image = None
        self.capper = cv2.VideoCapture('rtsp://192.168.1.3:5554') #cv2.VideoCapture(camera)

        self.resolution = (self.capper.get(cv2.CAP_PROP_FRAME_WIDTH),self.capper.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.current_detections = []

    def close(self):

        self.fetching = False
        self.conn.close()

    def start_fetching(self):

        threading.Thread(target=self.fetch_loop, args=(), daemon=True).start()

    def capture(self):

        _, self.image = self.capper.read()

    @property
    def image_bytes(self):
        _, encoded = cv2.imencode('.webp', self.image)
        return encoded.tobytes()

    @property
    def image_rgb(self):
        return numpy.rot90(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)) # RGB frame rotated as required by pygame

    def load_detections(self, raw_detections):

        detections = []
        detections_amount = len(raw_detections)//34 # 34 bytes per detection

        det_buffer = io.BytesIO(raw_detections)

        for n in range(detections_amount):

            classid = int.from_bytes(det_buffer.read(1), 'little')
            if classid not in class_mapping:
                det_buffer.seek(33, 1) # Skip rest of current detection if no class name set
                continue

            xmin = self.resolution[0] - int.from_bytes(det_buffer.read(8), 'little')
            ymin = int.from_bytes(det_buffer.read(8), 'little') # pygame y-coordinates start at top

            xmax = self.resolution[0] - int.from_bytes(det_buffer.read(8), 'little')
            ymax = int.from_bytes(det_buffer.read(8), 'little') # pygame y-coordinates start at top

            conf = int.from_bytes(det_buffer.read(1), 'little')

            detections.append({'class':class_mapping[classid], 'min':(xmin, ymin), 'max':(xmax, ymax), 'conf':conf})

        det_buffer.close()
        self.current_detections = detections

    def fetch_loop(self):

        try:
            while self.fetching:

                if not self.image is None:

                    self.conn.send(self.image_bytes)

                    detections_raw = self.conn.recv(2048)
                    if len(detections_raw) > 3:  # If more than standard 'DET' bytes
                        self.load_detections(detections_raw[3:]) # Load detections, skip leading 'DET' bytes
                    else:
                        self.current_detections = []

                time.sleep(self.wait_time)

        except ConnectionError:
            print('Connection to server lost.')
            sys.exit(1)
