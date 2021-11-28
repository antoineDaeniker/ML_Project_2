import torch
import os

# base path of the dataset
DATASET_PATH = os.path.join("datasets_full", "RPE1wt_CEP63+CETN2+PCNT_1", "projections_channel")

# define the path to the images and masks dataset
IMAGE_DATASET_PATH_NUCLEI = os.path.join(DATASET_PATH, "DAPI", "tif")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks") #TODO add path for ground thruth mask images

# define the path to the centriol data
DATASET_PATH_CENTRIOL_IMAGES = os.path.join(DATASET_PATH, "CEP63", "tif")
DATASET_PATH_CENTRIOL_MASKS = os.path.join("datasets_full", "annotations.csv") #TODO this is only centriols coords

# define the test split
TEST_SPLIT = 0.15 #TODO maybe we don't need this

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
#TODO understand these value
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 64 #TODO to be redefine

# define the input image dimensions
#TODO to be redefnine (maybe 256 x 256)
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128

# define threshold to filter weak predictions
THRESHOLD = 0.5 #TODO to understand and redefine

# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training
# plot, and testing image paths
#TODO to be redefine
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])