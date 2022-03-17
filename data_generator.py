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
from carla import VehicleLightState as vls
import random
# torchvision.models.detection.maskrcnn_resnet50_fpn
import time
import numpy as np
import cv2
import warnings
import matplotlib.pyplot as plt

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
SetVehicleLightState = carla.command.SetVehicleLightState
FutureActor = carla.command.FutureActor

# Import queue library
import queue

warnings.filterwarnings("ignore")
print("starting")
IM_WIDTH = 640
IM_HEIGHT = 480
FPS = 40
NUM_FRAMES = 300

def process_img(image):
    #model.eval()

    if(image is not None):
        start = time.time()
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        cv2.imshow('', i3)
        cv2.waitKey(5)
        #print("In the function")
        return i3/255.0

actor_list = []
try:
    frame_number = 0
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)

    world = client.get_world()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    # Control synchronous time
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0/FPS
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    # Adding other traffic
    other_vehicles = blueprint_library.filter('vehicle')

    other_vehicles = [x for x in other_vehicles if int(x.get_attribute('number_of_wheels')) == 4]
    other_vehicles = sorted(other_vehicles, key=lambda bp: bp.id)

    bp = blueprint_library.filter('model3')[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(bp, spawn_point)
    #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

    actor_list.append(vehicle) 

    for n, transform in enumerate(spawn_points):
        blueprint = random.choice(other_vehicles)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')
        # prepare the light state of the cars to spawn

        # spawn the cars and set their autopilot and light state all together
        actor_list.append(SpawnActor(blueprint, transform)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
            .then(SetVehicleLightState(FutureActor, light_state)))
    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            actor_list.append(response.actor_id)


    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
    camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
    camera_bp.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)

    # create queue to append sensor data
    camera_queue = queue.Queue()
    sensor.listen(camera_queue.put)
    world.tick()

    while (frame_number < NUM_FRAMES):
        vehicle.set_autopilot(True)
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

