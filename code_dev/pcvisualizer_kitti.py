# import os
# #import pcl
# import cv2
# import numpy as np
# from mayavi import mlab

# IMAGE_FILE_PATH = '/home/ubuntu/18744/Data/KITTI_mini/object/testing/image_2/000008.png'
# img = cv2.imread(IMAGE_FILE_PATH)
# pc = np.load("3d_data.npz")

# x = pc['xs']


# y = pc['ys']


# z = pc['zs']

# coords = []

# for i in range(len(x)):
# 	coords.append([x[i], y[i], z[i]])

# #p = pcl.PointCloud(np.array(coords))

# mlab.points3d(x, -y, z, -y, mode='point')
# mlab.xlabel('x')
# mlab.ylabel('y')
# mlab.zlabel('z')

# mlab.show()

# cv2.imshow(img)
# cv2.waitkey(5)

