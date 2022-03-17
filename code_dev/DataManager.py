import os
from datetime import datetime

class DataManager():
	def __init__(self, root_directory):
		date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
		self.capture_dir_name = f"capture_{date}"
		self.capture_path = os.path.join(root_directory, self.capture_dir_name)
		os.makedirs(self.capture_path)
	def get_capture_path(self):
		return self.capture_path