import numpy as np 
import config
from cv2 import cv2

def process_image(image, filename, show_image = True):
    i = np.array(image.raw_data)
    i2 = i.reshape((config.IMHEIGHT, config.IMWIDTH, 4))
    i3 = i2[:, :, :3]
    if(show_image == True):
    	cv2.imshow("", i3)
    	cv2.waitKey(5)
    return i3/255.0

