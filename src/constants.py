import torch
import os

# define the crop image dimensions before resizing
CROP_IMAGE_WIDTH = 16 # maybe 8
CROP_IMAGE_HEIGHT = 16 # maybe 8

# define the input image dimensions
INPUT_IMAGE_WIDTH = 32 # maybe 16
INPUT_IMAGE_HEIGHT = 32 # maybe 16

# full image dimension
FULL_IMAGE_WIDTH = 2048
FULL_IMAGE_HEIGHT = 2048

# number of input classes 
NUM_CLASSES = 2