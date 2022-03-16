import sys
import os
import config
sys.path.append(config.CARLA_EGG_PATH)

import carla 


class SimWorld:
	def __init__(self):
		client = carla.Client('127.0.0.1', 2000)
    	client.set_timeout(20.0)
    	self.world = client.get_world()

    def set_synchronous_mode(synch_mode = True, fps = 10, no_render = False):
    	settings = self.world.get_settings()
    	settings.synchronous_mode = True
    	settings.fixed_delta_seconds = 1.0/fps
    	settings.no_rendering_mode = False
    	self.world.apply_settings(settings)