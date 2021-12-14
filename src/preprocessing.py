import torch
from torchvision import transforms
import numpy as np
from scipy import signal

import constants
import utils

class RandomGaussianNoise(object):
    "Adds random gaussian noise to the image"
    
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        noise = torch.normal(mean=self.mean, std=self.std, size=image.shape)
        return (image + noise)


def standardize_patch(image):
    # standardize a tensor patch

    image_mean = torch.mean(image, dim=0)
    image_std = torch.std(image, dim=0) + 1e-10
    normalization_transform = transforms.Normalize(mean=image_mean, std=image_std)
    image = normalization_transform(image)
    return image


def normalize(image):
    # normalize each channel separately to [0, 1]

    normalized_image = np.zeros(image.shape, dtype=np.float)
    for i in range(image.shape[2]):
        min_val = np.min(image[:, :, i])
        max_val = np.max(image[:, :, i])
        range_val = max_val - min_val + 1e-10

        normalized_image[:, :, i] = np.array((image[:, :, i] - min_val) 
                                            / float(range_val), dtype=np.float)
    
    return normalized_image


def log_normalize(image):
    # normalize log of each channel separately to [0, 1]

    normalized_image = np.zeros(image.shape, dtype=np.float)
    for i in range(image.shape[2]):
        min_val = np.min(np.log(1 + image[:, :, i]))
        max_val = np.max(np.log(1 + image[:, :, i]))
        range_val = max_val - min_val + 1e-10

        normalized_image[:, :, i] = np.array((np.log(1 + image[:, :, i]) - min_val) 
                                            / float(range_val), dtype=np.float)

    return normalized_image


def thresholded_normalize(image, percentile=95):
    normalized_image = np.zeros(image.shape, dtype=np.float)
    for i in range(image.shape[2]):
        q = np.percentile(image[:, :, i], q=percentile)
        max_val = np.max(image[:, :, i])
        range_val = max_val - q + 1e-10

        normalized_image[image[:, :, i] < q, i] = -1
        normalized_image[image[:, :, i] >= q, i] = np.array((image[image[:, :, i] >= q, i] - q) 
                                                           / range_val, dtype=np.float)
    
    return normalized_image


def standardize(image):
    # standardize each channel separately

    standardized_image = np.zeros(image.shape, dtype=np.float)
    for i in range(image.shape[2]):
        mean = np.mean(image[:, :, i])
        std = np.std(image[:, :, i]) + 1e-10

        standardized_image[:, :, i] = np.array((image[:, :, i] - mean) / float(std), dtype=np.float)
    
    return standardized_image


def nonmaxima_suppression_box(image, size=3):
    domain = np.ones((size, size))
    max_val = signal.order_filter(image, domain, 8)
    image = np.where(image == max_val, image, 0)
    return image