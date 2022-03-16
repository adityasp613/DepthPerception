import numpy as np 
import scipy as sp
from cv2 import cv2

def process_image(image, filename, show_image = False):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    if(show_image == True):
    	cv2.imshow("", i3)
    	cv2.waitKey(5)
    #filename = f"/home/ubuntu/Pictures/image_{frame_id}.jpg"
    cv2.imwrite(filename, i3)
    return i3/255.0

