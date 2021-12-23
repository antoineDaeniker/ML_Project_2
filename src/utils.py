import torch
import numpy as np
import matplotlib.pyplot as plt

import constants



def set_seed(seed=44):
    """
    Set the seed for randomization

    Args:
        seed (int, optional): the seed value we want to set. Defaults to 44.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_image_mask(image, mask):
    """
    Plot image and mask for each channels

    Args:
        image : image to display
        mask : mask to display
    """
    channels = image.shape[2]

    if(channels == 1):
        fig, axs = plt.subplots(2, channels)
        axs[0].imshow(image[:, :, 0])
        axs[1].imshow(mask)
        plt.show()
    else:
        fig, axs = plt.subplots(2, channels)
        for i in range(channels):
            axs[0, i].imshow(image[:, :, i])
            axs[1, i].imshow(mask)
        plt.show()