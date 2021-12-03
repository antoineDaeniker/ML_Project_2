import constants
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from math import ceil


class SegmentationCentrioleTrain(Dataset):
	def __init__(self, image_paths, mask_paths, transform, num_samples=10):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.image_paths = image_paths
		self.mask_paths = mask_paths
		self.transform = transform
		self.num_samples = num_samples

	def __len__(self):
		# return number of samples per epoch
		return self.num_samples

	def __getitem__(self, idx):
		idx = idx % len(self.image_paths)

		# grab the image path from the current index
		image_path = self.image_paths[idx]

		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = cv2.imread(image_path)
#		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.mask_paths[idx], 0) #TODO idx is the "image_name" and we need to get the mask(for now we only have coordinates)

		# check to see if we are applying any transformations
		if self.transform is not None:
			# apply the transformations to both image and its mask
			image = self.transform(image)
			mask = self.transform(mask)

		"""
		norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
		image = norm(image)
		"""
		
		# return a tuple of the image and its mask
		return (image, mask)



class SegmentationCentrioleTest(Dataset):
	def __init__(self, image_paths, mask_paths, transform):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.image_paths = image_paths
		self.mask_paths = mask_paths
		self.transform = transform
		self.width_crops = ceil(constants.FULL_IMAGE_WIDTH / constants.INPUT_IMAGE_WIDTH)
		self.height_crops = ceil(constants.FULL_IMAGE_HEIGHT / constants.INPUT_IMAGE_HEIGHT)
		self.crops_per_image = self.width_crops * self.height_crops

		padded_size_width = self.width_crops * constants.INPUT_IMAGE_WIDTH - constants.FULL_IMAGE_WIDTH
		padded_size_height = self.height_crops * constants.INPUT_IMAGE_HEIGHT - constants.FULL_IMAGE_HEIGHT
		self.padding = (int(padded_size_width / 2), int(padded_size_height / 2))


	def __len__(self):
		# return the number of total samples contained in the dataset
		return 10
	#	return len(self.image_paths) * self.crops_per_image

	def __getitem__(self, idx):
		instance_idx = idx // self.crops_per_image
		idx = idx % self.crops_per_image

		idx_height = idx // self.width_crops
		idx = idx % self.width_crops

		idx_width = idx


		# grab the image path from the current index
		image_path = self.image_paths[instance_idx]

		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = cv2.imread(image_path) 
	#	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		mask = cv2.imread(self.mask_paths[instance_idx], 0) #TODO idx is the "image_name" and we need to get the mask(for now we only have coordinates)

		padding_transform = transforms.Compose([transforms.ToPILImage(),
                                          	    transforms.Pad(self.padding),
                                          	    transforms.ToTensor()])

		image = padding_transform(image)
		mask = padding_transform(mask)

		# check to see if we are applying any transformations
		if self.transform is not None:
			# apply the transformations to both image and its mask
			image = self.transform(image)
			mask = self.transform(mask)


		width_start = idx_width * constants.INPUT_IMAGE_WIDTH
		width_end = (idx_width + 1) * constants.INPUT_IMAGE_WIDTH

		height_start = idx_height * constants.INPUT_IMAGE_HEIGHT
		height_end = (idx_height + 1) * constants.INPUT_IMAGE_HEIGHT

		image = image[:, height_start:height_end, width_start:width_end]
		mask = mask[:, height_start:height_end, width_start:width_end]

		"""
		norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
		image = norm(image)
		"""

		# return a tuple of the image and its mask
		return (image, mask)