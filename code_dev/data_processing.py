import numpy as np 
import config
import os
from cv2 import cv2

def process_image(image, folder, frame_id, show_image = True):
    i = np.array(image.raw_data)
    i2 = i.reshape((config.IMHEIGHT, config.IMWIDTH, 4))
    i3 = i2[:, :, :3]
    if(show_image == True):
    	cv2.imshow("", i3)
    	cv2.waitKey(5)
    if(folder is not None):

        file_name = 'image_{}.jpg'.format(str(frame_id))
        file = os.path.join(folder, file_name)
        cv2.imwrite(file, i3)
    return i3/255.0

