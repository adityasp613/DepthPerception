import torch
import torchvision
from PIL import Image
import glob
import os
import sys
import time 
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
# torchvision.models.detection.maskrcnn_resnet50_fpn
import time
import numpy as np
import cv2
import warnings
import matplotlib.pyplot as plt

# Import queue library
import queue

warnings.filterwarnings("ignore")
print("starting")
IM_WIDTH = 640
IM_HEIGHT = 480
FPS = 40
NUM_FRAMES = 300

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

def process_img(image):
    #model.eval()

    if(image is not None):
        start = time.time()
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        dim = (300, 400)
        #i3 = np.resize(i3, (400, 300, 3))
        #resized = cv2.resize(i3, dim, interpolation = cv2.INTER_AREA)
        resized = np.transpose(i3, (2, 0, 1))
        rt = torch.from_numpy(resized)
        #rt = rt.reshape((3, 300, 400))
        #print(rt.size())
        #cv2.imshow("", i3)
        #cv2.waitKey(5)
        rt = rt.type(torch.FloatTensor)
        rt = rt/255.0
        #cv2.imwrite("/home/ubuntu/Pictures/imdata.jpg", i3)
        preds = model([rt])
        #print("Forward called")
        end = time.time()
        print("time = ", end-start)
       
        #print("In the function")
        return i3/255.0

actor_list = []
try:
    frame_number = 0
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)

    world = client.get_world()

    # Control synchronous time
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0/FPS
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

    actor_list.append(vehicle)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', str(IM_WIDTH))
    blueprint.set_attribute('image_size_y', str(IM_HEIGHT))
    blueprint.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)

    # create queue to append sensor data
    camera_queue = queue.Queue()
    sensor.listen(camera_queue.put)
    world.tick()

    while (frame_number < NUM_FRAMES):
        print(frame_number)
        frame_number += 1

        world.tick()
        if(not camera_queue.empty()):
            process_img(camera_queue.get())
            

    time.sleep(10)

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')

