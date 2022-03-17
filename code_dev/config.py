import carla
'''
Built-in weather primitives and choices

0: Default;
1: ClearNoon;
2: CloudyNoon;
3: WetNoon;
4: WetCloudyNoon;
5: MidRainyNoon;
6: HardRainNoon;
7: SoftRainNoon;
8: ClearSunset;
9: CloudySunset;
10: WetSunset;
11: WetCloudySunset;
12: MidRainSunset;
13: HardRainSunset;
14: SoftRainSunset;
15: ClearNight;
16: CloudyNight;
17: WetNight;
18: WetCloudyNight;
19: SoftRainNight;
20: MidRainyNight;
21: HardRainNight;

'''
SCENE_WEATHER = carla.WeatherParameters.ClearNoon

# Town for map load -> town0/7
WORLD_MAP = 'town02'

# Path to the CARLA egg file
CARLA_EGG_PATH = "/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.10-py3.6-linux-x86_64.egg"

# Simulation configuration
FPS = 40
NUM_FRAMES = 300
NUM_VEHICLES = 10

# Image dimensions
IMWIDTH = 640
IMHEIGHT = 480
IMFOV = 110

#Data save path
ROOT_DIR = '/home/ubuntu/18744/DepthPerception/carla_image_datasets'

