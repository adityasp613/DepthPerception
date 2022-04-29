''' Prepare KITTI data for 3D object detection.

Author: Charles R. Qi
Date: September 2017

Modified by Zhixin Wang
'''
import sys
sys.path.append("/home/ubuntu/18744/manucular_vision/DepthPerception/Code/frustum-convnet/")
#sys.path.append("/home/ubuntu/18744/manucular_vision/DepthPerception/Code/frustum-convnet/kitti/")

import argparse
import os
import pickle

import cv2
import numpy as np
from PIL import Image

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(BASE_DIR)
# sys.path.append(ROOT_DIR)


# import kitti_util as utils
# from kitti_object import kitti_object
# from draw_util import get_lidar_in_image_fov

from ops.pybind11.rbbox_iou import bbox_overlaps_2d

import configuration as config

class FrustumConvnet:

	def __init__(self):
		pass

	def get_lidar_in_image_fov(self, pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):

		pts_2d = pc_velo[:, :3] #calib.project_velo_to_image(pc_velo[:, :3])
		fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
			(pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
		fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
		imgfov_pc_velo = pc_velo[fov_inds, :]
		if return_more:
			return imgfov_pc_velo, pts_2d, fov_inds
		else:
			return imgfov_pc_velo

	def project_image_to_rect(self, uv_depth, calib_P2):
		''' Input: nx3 first two channels are uv, 3rd channel
		           is depth in rect camera coord.
		    Output: nx3 points in rect camera coord.
		'''

		# Camera intrinsics and extrinsics
		self.c_u = calib_P2[0, 2]
		self.c_v = calib_P2[1, 2]
		self.f_u = calib_P2[0, 0]
		self.f_v = calib_P2[1, 1]
		self.b_x = calib_P2[0, 3] / (-self.f_u)  # relative
		self.b_y = calib_P2[1, 3] / (-self.f_v)

		n = uv_depth.shape[0]
		x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
		y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
		pts_3d_rect = np.zeros((n, 3))
		pts_3d_rect[:, 0] = x
		pts_3d_rect[:, 1] = y
		pts_3d_rect[:, 2] = uv_depth[:, 2]

		return pts_3d_rect


	def prepare_data(self, det_id_list, det_type_list, det_box2d_list, det_prob_list, \
		output_filename, calib_P2, pc_velo, img_height_threshold=5, lidar_point_threshold=1):

		id_list = []
		type_list = []
		box2d_list = []
		prob_list = []
		input_list = []  # channel number = 4, xyz,intensity in rect camera coord
		frustum_angle_list = []  # angle of 2d box center from pos x-axis
		calib_list = []

		# calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
		# pc_velo = dataset.get_lidar(data_idx) #* 256 # image frame --> pc_velo

		pc_velo[:, 2] = pc_velo[:, 2] * 256
		pc_rect = np.zeros_like(pc_velo)

		pc_rect[:, 0:3] = self.project_image_to_rect(pc_velo[:, 0:3], calib_P2) # real world --> pc_rect

		pc_rect[:, 3] = pc_velo[:, 3]
		# img = dataset.get_image(data_idx)
		# img_height, img_width, img_channel = img.shape
		# print(img.shape)

		img_width = config.IMWIDTH
		img_height = config.IMHEIGHT

		_, pc_image_coord, img_fov_inds = self.get_lidar_in_image_fov(
		    pc_velo[:, 0:3], calib_P2, 0, 0, img_width, img_height, True)
		# cache = [calib, pc_rect, pc_image_coord, img_fov_inds]
		# cache_id = data_idx

		for det_idx in range(det_id_list):
		    # data_idx = det_id_list[det_idx]

		    # if det_type_list[det_idx] not in type_whitelist:
		    #     continue

		    # 2D BOX: Get pts rect backprojected
		    det_box2d = det_box2d_list[det_idx].copy()
		    det_box2d[[0, 2]] = np.clip(det_box2d[[0, 2]], 0, img_width - 1)
		    det_box2d[[1, 3]] = np.clip(det_box2d[[1, 3]], 0, img_height - 1)

		    xmin, ymin, xmax, ymax = det_box2d
		    box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
		                   (pc_image_coord[:, 0] >= xmin) & \
		                   (pc_image_coord[:, 1] < ymax) & \
		                   (pc_image_coord[:, 1] >= ymin)
		    box_fov_inds = box_fov_inds & img_fov_inds
		    pc_in_box_fov = pc_rect[box_fov_inds, :]

		    pc_box_image_coord = pc_image_coord[box_fov_inds, :]

		    # Get frustum angle (according to center pixel in 2D BOX)
		    box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
		    uvdepth = np.zeros((1, 3))
		    uvdepth[0, 0:2] = box2d_center
		    uvdepth[0, 2] = 20  # some random depth
		    box2d_center_rect = self.project_image_to_rect(uvdepth, calib_P2)
		    frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
		                                    box2d_center_rect[0, 0])

		    # Pass objects that are too small
		    if ymax - ymin < img_height_threshold or xmax - xmin < 1 or \
		            len(pc_in_box_fov) < lidar_point_threshold:
		        continue

		    id_list.append(det_idx)
		    type_list.append(det_type_list[det_idx])
		    box2d_list.append(np.array([xmin, ymin, xmax, ymax]))
		    prob_list.append(det_prob_list[det_idx])
		    input_list.append(pc_in_box_fov.astype(np.float32, copy=False))
		    frustum_angle_list.append(frustum_angle)
		    calib_list.append(calib_P2)

		# with open(output_filename, 'wb') as fp:
		#     pickle.dump(id_list, fp, -1)
		#     pickle.dump(box2d_list, fp, -1)
		#     pickle.dump(input_list, fp, -1)
		#     pickle.dump(type_list, fp, -1)
		#     pickle.dump(frustum_angle_list, fp, -1)
		#     pickle.dump(prob_list, fp, -1)
		#     pickle.dump(calib_list, fp, -1)

		# print('total_objects %d' % len(id_list))
		# print('save in {}'.format(output_filename))
	def test():
		pass
