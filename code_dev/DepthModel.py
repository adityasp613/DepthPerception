import numpy as np
import torch

class DepthModel:
	def __init__(self, model):
		self.transform = None
		self.device = None
		if(model == 'midas'):
			self.model_name = 'midas'
		elif(model == 'packnet'):
			self.model_name = 'packnet'
		else:
			self.model_name = 'midas'

	def configure_model(self):
		if(self.model_name == 'midas'):
			model_type = "DPT_Large"
			midas = torch.hub.load("intel-isl/MiDaS", model_type)
			self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
			midas.to(self.device)
			midas.eval()
			midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
			if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
				transform = midas_transforms.dpt_transform
			else:
				transform = midas_transforms.small_transform
			self.transform = transform
			self.model = midas
		elif(self.model_name == 'packnet'):
			return None
		else:
			return None

	def generate_depth_map(self, img):
		if(self.model_name == 'midas'):

			input_batch = self.transform(img).to(self.device)
			with torch.no_grad():
				prediction = self.model(input_batch)
				prediction = torch.nn.functional.interpolate(
				        prediction.unsqueeze(1),
				        size=img.shape[:2],
				        mode="bicubic",
				        align_corners=False,
				    ).squeeze()	
			depth = prediction.cpu().numpy()
			return depth
		elif(self.model_name == 'packnet'):
			return None
		else:
			return None
