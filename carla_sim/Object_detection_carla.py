import cv2
import numpy as np
import torch
import time

import carla
import pygame
from pygame.locals import *

# YOLOv5 imports
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import select_device

class CarlaGame(object):
    def __init__(self):
        pygame.init()
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []
        self._initialize_sensors()

        # YOLOv5 initialization
        self.device = select_device('')
        self.model = attempt_load('yolov5s.pt', map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def _initialize_sensors(self):
        # Set up the camera sensor
        cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '800')
        cam_bp.set_attribute('image_size_y', '600')
        cam_bp.set_attribute('fov', '110')
        cam_location = carla.Transform(carla.Location(x=1.5, z=2.4))
        cam_sensor = self.world.spawn_actor(cam_bp, cam_location)
        cam_sensor.listen(lambda data: self.process_image(data))

        # Set up the collision sensor
        col_bp = self.blueprint_library.find('sensor.other.collision')
        col_location = carla.Transform()
        col_sensor = self.world.spawn_actor(col_bp, col_location)
        col_sensor.listen(lambda event: self.process_collision(event))

        self.actor_list.append(cam_sensor)
        self.actor_list.append(col_sensor)

    def process_image(self, image):
        # Convert the image to a NumPy array
        image = np.frombuffer(image.raw_data, dtype=np.uint8)
        image = image.reshape((image.shape[0], image.shape[1], 4))
        image = image[:, :, :3]

        # Perform object detection using YOLOv5
        result = self.model(image, size=self.stride, augment=False)[0]
        result = non_max_suppression(result, conf_thres=0.5, iou_thres=0.5)

        # Process the detection results
        for det in result:
            if det is not None and len(det) > 0:
                det[:, :4] = scale_coords(image.shape[1:], det[:, :4], image.shape[:2]).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / self.stride).view(-1).tolist()
                    print(f'Detected {label} at {xywh}')

    def process_collision(self, event):
        # Process the collision event here
        # For example, you can print the collision information
        print(f"Collision with {event.other_actor.type_id}")

    def run(self):
        try:
            while True:
                self.world.tick()
                pygame.time.wait(10)

        finally:
            print('destroying actors')
            for actor in self.actor_list:
                actor.destroy()
            pygame.quit()


