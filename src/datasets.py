import torch
from torch.nn.functional import normalize
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import cv2
from math import ceil

import utils
import constants
import preprocessing


def image_mask_preprocessing(image_path, mask_path):
	# load the image from disk and read the associated mask from disk in grayscale mode
	image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
	mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
#	image = image.expand_dims(axis=-1)
	
	"""
	for i in range(image.shape[2]):
		image[:, :, i] = preprocessing.nonmaxima_suppression_box(image[:, :, i])
		kernel = np.ones((5, 5))
		image[:, :, i] = cv2.dilate(image[:, :, i], kernel=kernel, iterations=2)
		image = np.sum(image, axis=2)[:, :, np.newaxis]
	"""

	image = preprocessing.normalize(image)
#	image = preprocessing.standardize(image)
	return image, mask



class SegmentationCentrioleTrain(Dataset):
	def __init__(self, image_paths, mask_paths, transform=None, norm_transform=None, num_samples=5000):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.image_paths = image_paths
		self.mask_paths = mask_paths
		self.transform = transform
		self.norm_transform = norm_transform
		self.random_crop = transforms.RandomCrop(size=(constants.INPUT_IMAGE_WIDTH,
												       constants.INPUT_IMAGE_HEIGHT))
		self.num_samples = num_samples
		
		# preprocess all images for faster training
		self.image_masks = []
		for i in range(len(self.image_paths)):
			image, mask = image_mask_preprocessing(self.image_paths[i], self.mask_paths[i])

			# check to see if we are applying any transformations
			if self.transform is not None:
				image = self.transform(image)
				mask = self.transform(mask)

			if self.norm_transform is not None:
				image = self.norm_transform(image)

			# image and mask are concatenated for better performance
			image_mask = torch.cat((image, mask), dim=0)
			self.image_masks.append(image_mask)


	def __len__(self):
		# return number of samples per epoch
		return self.num_samples

	def __getitem__(self, idx):
		idx = idx % len(self.image_masks)

		image_mask = self.image_masks[idx]		

		image_mask = self.random_crop(image_mask)
		image = image_mask[:-1]
		mask = image_mask[[-1]]

		# standardize each patch separately
		image = preprocessing.standardize_patch(image)

		# return a tuple of the image and its mask
		return (image.float(), mask.long())



class SegmentationCentrioleTest(Dataset):
	def __init__(self, image_paths, mask_paths, transform=None, norm_transform=None):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.image_paths = image_paths
		self.mask_paths = mask_paths
		self.transform = transform
		self.norm_transform = norm_transform
		self.width_crops = ceil(constants.FULL_IMAGE_WIDTH / constants.INPUT_IMAGE_WIDTH)
		self.height_crops = ceil(constants.FULL_IMAGE_HEIGHT / constants.INPUT_IMAGE_HEIGHT)
		self.crops_per_image = self.width_crops * self.height_crops

		padded_size_width = self.width_crops * constants.INPUT_IMAGE_WIDTH - constants.FULL_IMAGE_WIDTH
		padded_size_height = self.height_crops * constants.INPUT_IMAGE_HEIGHT - constants.FULL_IMAGE_HEIGHT
		self.padding = (int(padded_size_width / 2), int(padded_size_height / 2))
		self.padding_transform = transforms.Pad(self.padding)

		# preprocess all images for faster training
		self.images = []
		self.masks = []
		for i in range(len(self.image_paths)):
			image, mask = image_mask_preprocessing(self.image_paths[i], self.mask_paths[i])

			# check to see if we are applying any transformations
			if self.transform is not None:
				# apply the transformations to both image and its mask
				image = self.transform(image)
				mask = self.transform(mask)

			image = self.padding_transform(image)
			mask = self.padding_transform(mask)

			if self.norm_transform is not None:
				image = self.norm_transform(image)

			self.images.append(image)
			self.masks.append(mask)

	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.image_paths) * self.crops_per_image

	def __getitem__(self, idx):
		instance_idx = idx // self.crops_per_image
		idx = idx % self.crops_per_image

		idx_height = idx // self.width_crops
		idx = idx % self.width_crops

		idx_width = idx

		image = self.images[instance_idx]
		mask = self.masks[instance_idx]

		width_start = idx_width * constants.INPUT_IMAGE_WIDTH
		width_end = (idx_width + 1) * constants.INPUT_IMAGE_WIDTH

		height_start = idx_height * constants.INPUT_IMAGE_HEIGHT
		height_end = (idx_height + 1) * constants.INPUT_IMAGE_HEIGHT

		image = image[:, height_start:height_end, width_start:width_end]
		mask = mask[:, height_start:height_end, width_start:width_end]

		# standardize patch separately
		image = preprocessing.standardize_patch(image)

		# return a tuple of the image and its mask
		return (image.float(), mask.long())