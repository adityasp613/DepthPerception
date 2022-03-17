import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
import config
import torch
import time
import os
from cv2 import cv2

def viz_inv_depth(inv_depth, normalizer=None, percentile=95,
              colormap='plasma', filter_zeros=False):
    """
    Converts an inverse depth map to a colormap for visualization.
    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map to be converted
    normalizer : float
        Value for inverse depth map normalization
    percentile : float
        Percentile value for automatic normalization
    colormap : str
        Colormap to be used
    filter_zeros : bool
        If True, do not consider zero values during normalization
    Returns
    -------
    colormap : np.array [H,W,3]
        Colormap generated from the inverse depth map
    """
    # If a tensor is provided, convert to numpy
    if torch.is_tensor(inv_depth):
        # Squeeze if depth channel exists
        if len(inv_depth.shape) == 3:
            inv_depth = inv_depth.squeeze(0)
        inv_depth = inv_depth.detach().cpu().numpy()
    cm = get_cmap(colormap)
    if normalizer is None:
        normalizer = np.percentile(
            inv_depth[inv_depth > 0] if filter_zeros else inv_depth, percentile)
    inv_depth /= (normalizer + 1e-6)
    return cm(np.clip(inv_depth, 0., 1.0))[:, :, :3]

def process_image(image, folder, frame_id, depth_model, show_image = True):
    i = np.array(image.raw_data)
    i2 = i.reshape((config.IMHEIGHT, config.IMWIDTH, 4))
    i3 = i2[:, :, :3]
    depth_map = depth_model.generate_depth_map(i3)
    # rgb = cv2.cvtColor(i3, cv2.COLOR_BGR2RGB)#np.transpose(i3, (1, 2, 0))
    # viz_pred_inv_depth = viz_inv_depth(depth_map) * 255
    # image_viz = np.concatenate([rgb, viz_pred_inv_depth], 0)
    depth_map_img = np.dstack((depth_map, depth_map, depth_map))
    depth_map_img = ((depth_map_img/255) * 255).astype('uint8')
    depth_map_img_gray = cv2.applyColorMap(depth_map_img, cv2.COLORMAP_JET)
    #depth_map_img_gray = cv2.cvtColor(depth_map_img, cv2.COLOR_BGR2RGB)
    # concat_image = np.concatenate((i3_gray, depth_map_img_gray), axis=1)
    if(show_image == True):
      
        cv2.imshow("Original image", i3)
        cv2.imshow("Depth map", depth_map_img_gray)
        cv2.waitKey(5)
    if(folder is not None):

        file_name = 'image_{}.jpg'.format(str(frame_id))
        file = os.path.join(folder, file_name)
        cv2.imwrite(file, i3)
    return i3/255.0

