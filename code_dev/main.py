import sys
import os
from SimWorld import SimWorld
import config

# Create a world object
carla_world = SimWorld()

print("Starting data capture")

carla_world.spawn_vehicles(config.NUM_VEHICLES)
carla_world.acquire_data(config.IMWIDTH, config.IMHEIGHT, config.IMFOV, config.FPS, config.NUM_FRAMES)