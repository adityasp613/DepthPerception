import os
import numpy as np
from mayavi import mlab

data_path = '/home/ubuntu/18744/DepthPerception/carla_image_datasets/capture_2022_03_18-09_31_47_PM'
file_name = 'pseudo_lidar_33.npy'

file_path = os.path.join(data_path, file_name)

subsample_factor = 1
pc = np.load(file_path)
z_th = 0.2
print(pc)
print(np.shape(pc))
x = pc[:,0]


y = pc[:,1]


z = pc[:,2]

indices = np.where(z < z_th)
print((np.max(z), np.min(z)))
xs = x[indices]
ys = y[indices]
zs = z[indices]

mlab.points3d(xs, -ys, zs, zs, mode='point')
mlab.xlabel('x')
mlab.ylabel('y')
mlab.zlabel('z')

mlab.show()
