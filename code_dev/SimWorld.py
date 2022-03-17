import sys
import os
import queue
import config
import random
import data_processing
sys.path.append(config.CARLA_EGG_PATH)

import carla 
#from set_synchronous_mode import CarlaSyncMode

class SimWorld:
	
	def __init__(self, scene, weather):
		self.client = carla.Client('127.0.0.1', 2000)
		self.client.set_timeout(20.0)
		self.world = self.client.load_world(scene)
		self.world.set_weather(weather)
		self.ego_vehicle = self.spawn_ego_vehicle()
		self.world.tick()
		self.vehicles_list = []
		self.sensors_list = []
		self.all_id = []
		self.batch = []
		

	def set_synchronous_mode(self, synch_mode = True, fps = 10, no_render = False):
		settings = self.world.get_settings()
		settings.synchronous_mode = True
		settings.fixed_delta_seconds = 1.0/fps
		settings.no_rendering_mode = False
		self.world.apply_settings(settings)

	def spawn_ego_vehicle(self):
		blueprint_library = self.world.get_blueprint_library()
		bp = blueprint_library.filter('model3')[0]
		spawn_point = random.choice(self.world.get_map().get_spawn_points())

		vehicle = self.world.try_spawn_actor(bp, spawn_point)

		while(vehicle is None):
			spawn_point = random.choice(self.world.get_map().get_spawn_points())
			vehicle = self.world.try_spawn_actor(bp, spawn_point)
			vehicle.set_autopilot(True)
		print('Ego vehicle spawned......')
		return vehicle



	def spawn_vehicles(self, num_vehicles):
		tm = self.client.get_trafficmanager(8000)
		tm.set_synchronous_mode(True)
		vehicle_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
		spawn_points = self.world.get_map().get_spawn_points()
		num_spawn_points = len(spawn_points)

		if num_vehicles < num_spawn_points:
			random.shuffle(spawn_points)
		else:
			num_vehicles = num_spawn_points

		SpawnActor = carla.command.SpawnActor
		SetAutopilot = carla.command.SetAutopilot
		FutureActor = carla.command.FutureActor

	    # --------------
	    # Spawn vehicles
	    # --------------
		for n, transform in enumerate(spawn_points):
			if n >= num_vehicles:
			    break
			blueprint = random.choice(vehicle_blueprints)
			# Taking out bicycles and motorcycles, since the semantic/bb labeling for that is mixed with pedestrian
			if int(blueprint.get_attribute('number_of_wheels')) > 2:
			    if blueprint.has_attribute('color'):
			        color = random.choice(blueprint.get_attribute('color').recommended_values)
			        blueprint.set_attribute('color', color)
			    if blueprint.has_attribute('driver_id'):
			        driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
			        blueprint.set_attribute('driver_id', driver_id)
			    blueprint.set_attribute('role_name', 'autopilot')
			    self.batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

		for response in self.client.apply_batch_sync(self.batch):
		    if response.error:
		        print("Error in setting batch sync")
		    else:
		        self.vehicles_list.append(response.actor_id)
		self.world.tick()
		return self.vehicles_list

	def configure_camera(self, vehicle, cam_imwidth, cam_imheight, cam_fov, x_loc, z_loc, queue):
		cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
		cam_bp.set_attribute('image_size_x', f'{cam_imwidth}')
		cam_bp.set_attribute('image_size_y', f'{cam_imheight}')
		cam_bp.set_attribute('fov', f'{cam_fov}')

		# spawn_point = carla.Transform(carla.Location(x=x_loc,z = z_loc))
		# self.rgb_camera = self.world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
		transform = carla.Transform(carla.Location(x=0.8, z=1.7))
		self.rgb_camera = self.world.spawn_actor(cam_bp, transform, attach_to = vehicle)
		self.rgb_camera.blur_amount = 0.0
		self.rgb_camera.motion_blur_intensity = 0
		self.rgb_camera.motion_max_distortion = 0
		self.rgb_camera.listen(queue.put)
		
		self.sensors_list.append(self.rgb_camera)


		return self.rgb_camera

	def acquire_data(self, imwidth, imheight, imfov, frame_rate, num_frames):
		current_frame = 0 
		camera_queue = queue.Queue();

		print(self.world.get_actors().filter("vehicle.*"))
		#my_vehicle = self.world.get_actors().filter("vehicle.*")[0]#random.choice([x for x in self.world.get_actors().filter("vehicle.*") if x.type_id not in
                                    # ['vehicle.audi.tt', 'vehicle.carlamotors.carlacola', 'vehicle.volkswagen.t2']])
		
		self.configure_camera(self.ego_vehicle, imwidth, imheight, imfov, 2.5, 0.7, camera_queue)

		
		while (current_frame< num_frames):
			print(current_frame)
			current_frame += 1

			self.world.tick()
			if(not camera_queue.empty()):
				self.client.apply_batch([carla.command.SetAutopilot(x, True) for x in [v for v in self.world.get_actors().filter("vehicle.*")]])
				self.ego_vehicle.set_autopilot(True)
				data_processing.process_image(camera_queue.get(), 'dummy')
