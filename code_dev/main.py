import sys
import os
from DepthModel import DepthModel
from DataManager import DataManager
from SimWorld import SimWorld
import config

#Create data directory
data_manager = DataManager(config.ROOT_DIR)
capture_path = data_manager.get_capture_path()

#Depth map generation
depth_generator = DepthModel(config.DEPTH_MODEL)
depth_generator.configure_model()
# Create a world object
carla_world = SimWorld(config.WORLD_MAP, config.SCENE_WEATHER)

print("Starting data capture")
carla_world.set_synchronous_mode(synch_mode = True, fps = config.FPS, no_render = False)
vehicles = carla_world.spawn_vehicles(config.NUM_VEHICLES)
print(len(vehicles))
carla_world.acquire_data(config.IMWIDTH, config.IMHEIGHT, config.IMFOV, config.FPS, 
						config.NUM_FRAMES, depth_generator, capture_path)