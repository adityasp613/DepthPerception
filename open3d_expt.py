import open3d as o3d 
import cv2
import numpy as np
print(o3d.__file__)
print("Read Redwood dataset")
redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()
print(np.shape(redwood_rgbd))

x = redwood_rgbd.color_paths[0]
depth_raw = o3d.io.read_image(redwood_rgbd.depth_paths[0])

print(np.shape(depth_raw))
img = o3d.io.read_image(x)
print(type(np.array(img)))
print(np.shape(img))
cv2.imshow('', np.array(img))
cv2.waitKey(0)