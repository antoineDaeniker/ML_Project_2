import torch
import os


# define the input image dimensions
INPUT_IMAGE_WIDTH = 32 # maybe 16
INPUT_IMAGE_HEIGHT = 32 # maybe 16

# full image dimension
FULL_IMAGE_WIDTH = 2048
FULL_IMAGE_HEIGHT = 2048

# TODO: WE HAVE TO INCREASE IF WE JOINTLY SEGMENT NUCLEI AND CENTRIOLES
# number of input classes 
NUM_CLASSES = 2


"""

# base path of the dataset
DATASET_PATH = os.path.join("datasets_full", "RPE1wt_CEP63+CETN2+PCNT_1", "projections_channel")

# define the path to the images and masks dataset
IMAGE_DATASET_PATH_NUCLEI = os.path.join(DATASET_PATH, "DAPI", "tif")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks") #TODO add path for ground thruth mask images

# define the path to the centriol data
DATASET_PATH_CENTRIOL_IMAGES = os.path.join(DATASET_PATH, "CEP63", "tif")
DATASET_PATH_CENTRIOL_MASKS = os.path.join("datasets_full", "annotations.csv") #TODO this is only centriols coords




# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training
# plot, and testing image paths
#TODO to be redefine
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
"""