import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import numpy as np
import cv2
from math import ceil

import utils
import constants
import preprocessing


def image_mask_preprocessing(image_path, mask_path, dataset_config):
	"""
	Load and pre-process images and mask according to dataset_config

	Args:
		image_path (string): the path to images dataset folder
		mask_path (string): the path to gt mask dataset folder
		dataset_config : configs for loading dataset

	Returns:
		image : preprocessed image
		mask : mask
	"""
	# load the image from disk and read the associated mask from disk in grayscale mode
	image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(dtype=np.float)
	mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
	
	if(dataset_config["channels"] == 1):
		image = np.expand_dims(image, axis=-1)

	if(dataset_config["nonmaxima_suppresion"]):
		for i in range(image.shape[2]):
			# thresholding
			q = np.percentile(image[:, :, i], q=98.5)
			image[:, :, i][image[:, :, i] < q] = 0

			# nonmaxima suppresion
			image[:, :, i] = preprocessing.nonmaxima_suppression_box(image[:, :, i])
			kernel = np.ones((3, 3))
			image[:, :, i] = cv2.dilate(image[:, :, i], kernel=kernel, iterations=1)
	
	if(dataset_config["normalize"]):
		image = preprocessing.normalize(image)

	if(dataset_config["standardize"]):
		image = preprocessing.standardize(image)

	return image, mask



class SegmentationCentrioleTrain(Dataset):
	"""
	Represent our centriol train dataset for the segmetation model

	Args:
		Dataset : centriol dataset
	"""
	def __init__(self, image_paths, mask_paths, dataset_config, transform=None, data_augmentation=None):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.image_paths = image_paths
		self.mask_paths = mask_paths
		self.transform = transform
		min_pos_p = dataset_config["min_positive_prob"]

		# data augmentation should be dictonary with two transforms:
		# one which is applied only to the image (gaussian blur) 
		# and one which is applied to both of them simultaneously (random flip, rotation, ...)
		self.data_augmentation = data_augmentation
		self.random_crop = transforms.RandomCrop(size=(constants.CROP_IMAGE_WIDTH,
												       constants.CROP_IMAGE_HEIGHT))
		self.resize_transform = transforms.Resize(size=(constants.INPUT_IMAGE_HEIGHT,
						  								constants.INPUT_IMAGE_WIDTH),
												  interpolation=InterpolationMode.NEAREST)
		
		# preprocess all images for faster training
		self.images = []
		self.masks = []
		for i in range(len(self.image_paths)):
			image, mask = image_mask_preprocessing(self.image_paths[i], self.mask_paths[i], dataset_config)

			# store only crops that contain at least min_pos_p of positive pixels
			kernel = np.ones((constants.CROP_IMAGE_WIDTH, constants.CROP_IMAGE_HEIGHT))
			number_of_positive = cv2.filter2D(src=mask / 255, ddepth=-1, kernel=kernel)
			center_indices = np.argwhere(number_of_positive > 
										(kernel.shape[0] * kernel.shape[1]) * min_pos_p)
			half_dim = constants.CROP_IMAGE_HEIGHT / 2

			# check to see if we are applying any transformations
			if self.transform is not None:
				image = self.transform(image)
				mask = self.transform(mask)

			# save crops with enough positive pixels
			for i in range(center_indices.shape[0]):
				# positive example
				x, y = center_indices[i]
				top, left = int(x - half_dim + 1), int(y - half_dim + 1)
				image_crop = transforms.functional.crop(image, top, left, constants.CROP_IMAGE_HEIGHT, 
																		  constants.CROP_IMAGE_WIDTH)
			
				mask_crop = transforms.functional.crop(mask, top, left, constants.CROP_IMAGE_HEIGHT, 
														                constants.CROP_IMAGE_WIDTH)
				
				if(mask_crop.shape[1] == constants.CROP_IMAGE_HEIGHT and 
					  mask_crop.shape[2] == constants.CROP_IMAGE_WIDTH):
						image_crop = self.resize_transform(image_crop)
						mask_crop = self.resize_transform(mask_crop)
						self.images.append(image_crop)	   
						self.masks.append(mask_crop)


				# negative example with probability min_pos_p
				if(np.random.uniform() < min_pos_p):
					x = np.random.randint(constants.CROP_IMAGE_HEIGHT)
					y = np.random.randint(constants.CROP_IMAGE_WIDTH)
					top, left = int(x - half_dim + 1), int(y - half_dim + 1)
					image_crop = transforms.functional.crop(image, top, left, constants.CROP_IMAGE_HEIGHT, 
																			constants.CROP_IMAGE_WIDTH)
				
					mask_crop = transforms.functional.crop(mask, top, left, constants.CROP_IMAGE_HEIGHT, 
																			constants.CROP_IMAGE_WIDTH)
					
					if(mask_crop.shape[1] == constants.CROP_IMAGE_HEIGHT and 
					  mask_crop.shape[2] == constants.CROP_IMAGE_WIDTH):
						image_crop = self.resize_transform(image_crop)
						mask_crop = self.resize_transform(mask_crop)
						self.images.append(image_crop)	   
						self.masks.append(mask_crop)


	def __len__(self):
		# return number of samples per epoch
		return len(self.images)

	def __getitem__(self, idx):
		image = self.images[idx]
		mask = self.masks[idx]

		if self.data_augmentation is not None and self.data_augmentation["both"] is not None:
			image_mask = torch.cat((image, mask), dim=0)
			image_mask = self.data_augmentation["both"](image_mask)
			image = image_mask[:-1]
			mask = image_mask[[-1]]

		if self.data_augmentation is not None and self.data_augmentation["image"] is not None:
			image = self.data_augmentation["image"](image)

		# return a tuple of the image and its mask
		return (image.float(), mask.long())



class SegmentationCentrioleTest(Dataset):
	"""
	Represent our centriol test dataset for the segmetation model

	Args:
		Dataset : centriol dataset
	"""
	def __init__(self, image_paths, mask_paths, dataset_config, transform=None):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.image_paths = image_paths
		self.mask_paths = mask_paths
		self.transform = transform
		self.resize_transform = transforms.Resize(size=(constants.INPUT_IMAGE_HEIGHT,
						  								constants.INPUT_IMAGE_WIDTH),
												  interpolation=InterpolationMode.NEAREST)

		self.width_crops = ceil(constants.FULL_IMAGE_WIDTH / constants.CROP_IMAGE_WIDTH)
		self.height_crops = ceil(constants.FULL_IMAGE_HEIGHT / constants.CROP_IMAGE_HEIGHT)
		self.crops_per_image = self.width_crops * self.height_crops
		

		padded_size_width = self.width_crops * constants.CROP_IMAGE_WIDTH - constants.FULL_IMAGE_WIDTH
		padded_size_height = self.height_crops * constants.CROP_IMAGE_HEIGHT - constants.FULL_IMAGE_HEIGHT
		self.padding = (int(padded_size_width / 2), int(padded_size_height / 2))
		self.padding_transform = transforms.Pad(self.padding)

		# preprocess all images for faster training
		self.images = []
		self.masks = []
		for i in range(len(self.image_paths)):
			image, mask = image_mask_preprocessing(self.image_paths[i], self.mask_paths[i], dataset_config)

			# check to see if we are applying any transformations
			if self.transform is not None:
				# apply the transformations to both image and its mask
				image = self.transform(image)
				mask = self.transform(mask)

			image = self.padding_transform(image)
			mask = self.padding_transform(mask)

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

		width_start = idx_width * constants.CROP_IMAGE_WIDTH
		width_end = (idx_width + 1) * constants.CROP_IMAGE_WIDTH

		height_start = idx_height * constants.CROP_IMAGE_HEIGHT
		height_end = (idx_height + 1) * constants.CROP_IMAGE_HEIGHT

		image = image[:, height_start:height_end, width_start:width_end]
		mask = mask[:, height_start:height_end, width_start:width_end]
		
		image = self.resize_transform(image)
		mask = self.resize_transform(mask)

		# return a tuple of the image and its mask
		return (image.float(), mask.long())