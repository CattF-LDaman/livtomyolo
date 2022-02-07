
import socket
import threading
import sys

from AppTest import HNDSHK
from .detect import detect

class Server:

    def __init__(self, port=7563):

        self.socket = socket.socket()
        self.bind_socket(port)

        self.accept_connections()

    def bind_socket(self, port):
        try:
            self.socket.bind(("", port))
            print(f"Socket bound on port {port}")
        except OSError:
            print(f"Couldn't bind socket to port {port}, maybe another server instance is already running.")
            sys.exit(1)

    def serve_connection(self, conn, addr):

        handshake = conn.recv(len(HNDSHK))
        if handshake != HNDSHK:
            print(f"{addr[0]}:{addr[1]} failed the handshake. Aborting connection.")
            conn.close()
            return

        try:

            while True:

                img = conn.recv(1024*1024*2)
                if not img:
                    print(f'Connection closed by {addr[0]}:{addr[1]}')
                    return

                detections = detect(img)
                conn.send(detections)

        except ConnectionError:
            print(f'Connection lost with {addr[0]}:{addr[1]}')

    def accept_connections(self):

        self.socket.listen()
        print("Started listening on socket")

        print("Connections will now be accepted!")

        while True:
            conn, addr = self.socket.accept()

            print(f"New connection with {addr[0]}:{addr[1]}")

            threading.Thread(target=self.serve_connection, args=(conn,addr), daemon=True).start() # Start serving in new thread
