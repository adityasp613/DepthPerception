import carla


SCENE_WEATHER = 'morning'

# Town for map load -> town0/7
WORLD_MAP = 'town07'

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

PACKNET_CHECKPOINT_FILE = '/home/ubuntu/18744/DepthPerception/code_dev/PackNet01_MR_selfsup_D.ckpt'

# Flag to display or not display image
SHOW_IMAGE = True
