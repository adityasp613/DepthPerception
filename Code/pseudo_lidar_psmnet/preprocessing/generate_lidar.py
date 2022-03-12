import argparse
import os

import numpy as np
import scipy.misc as ssc

import kitti_util

import matplotlib.pyplot as plt


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Libar')
    parser.add_argument('--calib_dir', type=str,
                        default='~/Kitti/object/training/calib')
    parser.add_argument('--disparity_dir', type=str,
                        default='~/Kitti/object/training/predicted_disparity')
    parser.add_argument('--save_dir', type=str,
                        default='~/Kitti/object/training/predicted_velodyne')
    parser.add_argument('--save_plot_dir', type=str,
                        default='~/Kitti/object/training/pseudo_lidar_plot')
    parser.add_argument('--max_high', type=int, default=1)
    parser.add_argument('--is_depth', action='store_true')
    parser.add_argument('--save_plot', type=bool, default=False)

    args = parser.parse_args()

    assert os.path.isdir(args.disparity_dir)
    assert os.path.isdir(args.calib_dir)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    disps = [x for x in os.listdir(args.disparity_dir) if x[-3:] == 'png' or x[-3:] == 'npy']
    disps = sorted(disps)

    for fn in disps:
        predix = fn[:-4]
        calib_file = '{}/{}.txt'.format(args.calib_dir, predix)
        calib = kitti_util.Calibration(calib_file)
        # disp_map = ssc.imread(args.disparity_dir + '/' + fn) / 256.
        if fn[-3:] == 'png':
            disp_map = ssc.imread(args.disparity_dir + '/' + fn)
        elif fn[-3:] == 'npy':
            disp_map = np.load(args.disparity_dir + '/' + fn)
        else:
            assert False
        if not args.is_depth:
            disp_map = (disp_map*256).astype(np.uint16)/256.
            lidar = project_disp_to_points(calib, disp_map, args.max_high)
        else:
            disp_map = (disp_map).astype(np.float32)/256.
            pseudo_lidar, lidar = project_depth_to_points(calib, disp_map, args.max_high)
        # pad 1 in the indensity dimension
        lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
        lidar = lidar.astype(np.float32)
        # lidar.tofile('{}/{}.bin'.format(args.save_dir, predix))
        # print('Finish Depth {}'.format(predix))

        #3D scatter plot of pseudo lidar point cloud
        if args.save_plot:
          fig = plt.figure()
          ax = fig.add_subplot(projection='3d')

          xs = pseudo_lidar[:, 0] * 255
          ys = pseudo_lidar[:, 1] * 255
          zs = pseudo_lidar[:, 2] * 255

          ax.scatter(xs, ys, zs)
          pseudo_lidar = np.reshape(pseudo_lidar, (disp_map.shape[0], disp_map.shape[1], 3))
          fig.savefig('{}/{}.png'.format(args.save_plot_dir, predix))

        np.save('{}/{}.npy'.format(args.save_dir, predix), pseudo_lidar)
        print('Finish Depth {}'.format(predix))

