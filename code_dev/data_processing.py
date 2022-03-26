import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
import configuration as config
import torch
import time
from cv2 import cv2

sys.path.append("/home/ubuntu/18744/DepthPerception/Code/packnet-sfm/")
from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor

computation_time_list = []

def project_disp_to_points(calib, disp, max_high):
    disp[disp < 0] = 0
    baseline = 0.54
    mask = disp > 0
    depth = calib.f_u * baseline / (disp + 1. - mask)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]

def project_depth_to_points(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    pseudo_cloud_rect = calib.project_image_to_rect(points)
    cloud = calib.project_rect_to_velo(pseudo_cloud_rect)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return pseudo_cloud_rect, cloud[valid]

@torch.no_grad()
def infer_and_save_depth(input_file, output_file, model_wrapper, image_shape, half, save):
    """
    Process a single input file to produce and save visualization

    Parameters
    ----------
    input_file : str
        Image file
    output_file : str
        Output file, or folder where the output will be saved
    model_wrapper : nn.Module
        Model wrapper used for inference
    image_shape : Image shape
        Input image shape
    half: bool
        use half precision (fp16)
    save: str
        Save format (npz or png)
    """
    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(output_file, os.path.basename(input_file))

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image = load_image(input_file)
    # Resize and to tensor
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to('cuda:{}'.format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image)['inv_depths'][0]

    if save == 'npz' or save == 'png' or save == 'npy':
        # Get depth from predicted depth map and save to different formats
        filename = '{}.{}'.format(os.path.splitext(output_file)[0], save)
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(filename, 'magenta', attrs=['bold'])))
        depth_npy = inv2depth(pred_inv_depth)
        write_depth(filename, depth=inv2depth(pred_inv_depth))
        return depth_npy
    else:
        # Prepare RGB image
        rgb = image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # Prepare inverse depth
        viz_pred_inv_depth = viz_inv_depth(pred_inv_depth[0]) * 255
        # Concatenate both vertically
        image = np.concatenate([rgb, viz_pred_inv_depth], 0)
        # Save visualization
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(output_file, 'magenta', attrs=['bold'])))
        imwrite(output_file, image[:, :, ::-1])
        return image[:, :, ::-1]

def project_image_to_rect(uv_depth, calibration):
    ''' Input: nx3 first two channels are uv, 3rd channel
               is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    '''
    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - calibration[0, 2]) * uv_depth[:, 2]) / calibration[0, 0] + calibration[0, 3]
    y = ((uv_depth[:, 1] - calibration[1, 2]) * uv_depth[:, 2]) / calibration[1, 1] + calibration[1, 3]
    pts_3d_rect = np.zeros((n, 3))
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    return pts_3d_rect

def generate_point_cloud(depth, camera_matrix):
    # print(np.shape(depth))
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    pseudo_cloud_rect = project_image_to_rect(points, camera_matrix)
    return pseudo_cloud_rect

def process_image(image, folder, frame_id, depth_model, calibration, show_image = False):
    # print("Calibration matrix")
    # print(calibration)
    i = np.array(image.raw_data)
    i2 = i.reshape((config.IMHEIGHT, config.IMWIDTH, 4))
    i3 = i2[:, :, :3]
    start_time = time.time()
    depth_map, depth_viz = depth_model.generate_depth_map(i3)
    end_time = time.time()
    computation_time_list.append(end_time - start_time)
    average_inference_time = np.mean(computation_time_list)
    print("Running average inference time = ", average_inference_time)
    disp_map = (depth_map).astype(np.float32)/256
    point_cloud = generate_point_cloud(disp_map, calibration)
    #print(point_cloud)
    if(config.DEPTH_MODEL == 'packnet'):
        depth_viz= ((depth_viz/255) * 255).astype('uint8')
        depth_map_img = cv2.cvtColor(depth_viz, cv2.COLOR_RGB2BGR)
        depth_map_img_gray = cv2.applyColorMap(depth_map_img, cv2.COLORMAP_MAGMA)
    elif(config.DEPTH_MODEL == 'midas'):
        depth_map_img = np.dstack((depth_map, depth_map, depth_map))
        depth_map_img = ((depth_map_img/255) * 255).astype('uint8')
        depth_map_img_gray = cv2.applyColorMap(depth_map_img, cv2.COLORMAP_HSV)
    else:
        pass
    if(show_image == True):
        #pass
        cv2.imshow("Original image", i3)
        cv2.imshow("Depth map", depth_map_img_gray)
        cv2.waitKey(5)
    if(folder is not None):
        depth_file = 'depth_map_{}.jpg'.format(str(frame_id))
        lidar_file = 'pseudo_lidar_{}'.format(str(frame_id))
        file_name = 'image_{}.jpg'.format(str(frame_id))
        file = os.path.join(folder, file_name)
        lidar_path = os.path.join(folder, lidar_file)
        depth_path = os.path.join(folder, depth_file)
        np.save(lidar_path, point_cloud)
        cv2.imwrite(file, i3)
        cv2.imwrite(depth_path, depth_map_img_gray)
    return i3/255.0

def process_depth(image, folder, frame_id, depth_model, calibration, show_image = True):
    data = np.array(image.raw_data)
    data = data.reshape((config.IMHEIGHT, config.IMWIDTH, 4))
    data = data.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(data[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    depth_meters = normalized_depth * 1000
    print("Shape of depth map is: ", np.shape(depth_meters))
    return depth_meters