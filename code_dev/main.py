import sys
import os
from SimWorld import SimWorld
import config

# Create a world object
carla_world = SimWorld(config.WORLD_MAP, config.SCENE_WEATHER)

print("Starting data capture")
carla_world.set_synchronous_mode(synch_mode = True, fps = config.FPS, no_render = False)
vehicles = carla_world.spawn_vehicles(config.NUM_VEHICLES)
print(len(vehicles))
carla_world.acquire_data(config.IMWIDTH, config.IMHEIGHT, config.IMFOV, config.FPS, config.NUM_FRAMES)