
from skeleton_functions import *

def skeleton_extractor(directories_path="Dataset/PaddedFrames"):
    frame_to_process = 75
    directories = os_sorted(os.listdir(directories_path))
    for directory in directories:
        for frame in range(0, len(os.listdir(directories_path + "/" + directory)), frame_to_process):
            os.system("python skeleton_functions.py --directory=" + directory + " --frame=" + str(frame))
        normalize_directory(directory)


skeleton_extractor()




