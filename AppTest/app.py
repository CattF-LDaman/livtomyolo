
from .fetcher import Fetcher
from .dataset import class_color_mapping

from contextlib import redirect_stdout
with redirect_stdout(None):
    import pygame
    
import sys

class App:

    def __init__(self, server_address, server_port=7563):

        self.fetcher = Fetcher(server_address, server_port)
        print('Fetcher created.')

        self.screen = pygame.display.set_mode(self.fetcher.resolution)
        pygame.display.set_caption(f"Object Detection: {server_address}:{server_port}")

        self.running = True
        self.fetcher.start_fetching()
        self.loop()

    def handle_events(self):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.fetcher.close()
                self.running = False
                sys.exit()

    def draw(self):

        if self.fetcher.image is None: # Fill screen with red if fetcher has no image
            self.screen.fill((255,0,0))
            return

        img_surf = pygame.surfarray.make_surface(self.fetcher.image_rgb)
        self.screen.blit(img_surf, (0,0))

        for detection in self.fetcher.current_detections:
            det_rect = pygame.Rect(0,0,0,0)
            det_rect.height = detection['max'][1] - detection['min'][1]
            det_rect.bottomright = detection['max'][0], detection['max'][1]
            det_rect.width = detection['min'][0] - detection['max'][0]

            pygame.draw.rect(self.screen, class_color_mapping[detection['class']], det_rect, 1)

    def loop(self):

        while self.running:

            self.fetcher.capture()

            self.handle_events()
            self.draw()

            pygame.display.flip() # Update window
