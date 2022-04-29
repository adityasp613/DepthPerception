import carla


SCENE_WEATHER = 'morning'

# Town for map load -> town0/7
WORLD_MAP = 'town04'

# Path to the CARLA egg file
CARLA_EGG_PATH = "/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.10-py3.6-linux-x86_64.egg"

# Simulation configuration
FPS = 40
NUM_FRAMES = 1000
IGNORE_NUM_FRAMES = 100
NUM_VEHICLES = 100

# Image dimensions
IMWIDTH = 640
IMHEIGHT = 480
IMFOV = 90

#Data save path
ROOT_DIR = '/home/ubuntu/18744/DepthPerception/carla_image_datasets'

# Choice of depth model
DEPTH_MODEL = 'packnet'

PACKNET_CHECKPOINT_FILE = '/home/ubuntu/18744/manucular_vision/epoch_25.ckpt'

# Flag to display or not display image
SHOW_IMAGE = True

SEGMENTATION_NETWORK = 'detectron'
pred_classes_mapping = ['background',
 'person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush']