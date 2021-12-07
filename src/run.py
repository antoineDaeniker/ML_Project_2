import os
import sys
import glob
import json
import argparse

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import constants
import utils
import preprocessing

from datasets import SegmentationCentrioleTrain, SegmentationCentrioleTest
from model import UNet
from train import train



def arg_parse():
    parser = argparse.ArgumentParser(description='Centriole segmentation arguments.')

    parser.add_argument("--config", type=str, required=True, 
                        help="The config file that contains all hyperparameters")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='CPU / GPU device.')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='CPU / GPU device.')
    parser.add_argument('--dataset_dir', type=str, default="../datasets_full/all_channel_img",
                        help='Dataset directory.')
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints",
                        help='Directory for saving checkpoints.')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Whether to use wandb logging.')

    return parser.parse_args()



if __name__ == '__main__':
    _root_path = os.path.join(os.path.dirname(__file__) , '..')
    sys.path.insert(0, _root_path)

    args = arg_parse()

    assert os.path.exists(args.config), f"File {args.config} does not exist!"
    config = json.load(open(args.config, "r"))
    utils.set_seed(config["seed"])


    args.device = (args.device if torch.cuda.is_available() else 'cpu')
    args.pin_memory = True if args.device != "cpu" else False


    image_format = "tif"
    image_path = f"{args.dataset_dir}/{image_format}"
    mask_path = f"{args.dataset_dir}/masks"
    
    # relative paths to images
    all_images = []
    for image in glob.glob(f"{image_path}/*.{image_format}"):
        all_images.append(image)
        print(image)

    # train/val/test split: 80/10/10
    train_images, val_images = train_test_split(all_images, test_size=0.2)
    val_images, test_images = train_test_split(val_images, test_size=0.5)

    train_masks = [f'{mask_path}/{image.split("/")[-1][:-4]}_mask.png' for image in train_images] 
    val_masks = [f'{mask_path}/{image.split("/")[-1][:-4]}_mask.png' for image in val_images] 
    test_masks = [f'{mask_path}/{image.split("/")[-1][:-4]}_mask.png' for image in test_images] 

   
    print(f"Training set: {len(train_images)} instances.")
    print(f"Validation set: {len(val_images)} instances.")
    print(f"Test set: {len(test_images)} instances.")
   
    data_augmentation = {}
    data_augmentation["both"] = None
    data_augmentation["image"] = None

    """
    data_augmentation = {}
    data_augmentation["both"] = transforms.Compose([
                                                    transforms.RandomHorizontalFlip(p=0.3),
                                                    transforms.RandomVerticalFlip(p=0.3),
                                                    transforms.RandomApply(
                                                         transforms.RandomRotation(degrees=40), p=0.6
                                                        )
                                                   ])
    # Maybe also transforms.RandomResizedCrop???
    data_augmentation["image"] = transforms.Compose([transforms.RandomGrayscale(p=0.1),
                                                     transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 2)),
                                                     preprocessing.RandomGaussianNoise(mean=0, std=1)
                                                    ])
    """
    
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])


    train_dataset = SegmentationCentrioleTrain(train_images, train_masks, 
                                               transform=train_transform,
                                               dataset_config=config["dataset"])
    val_dataset = SegmentationCentrioleTest(val_images, val_masks, 
                                            transform=test_transform,
                                            dataset_config=config["dataset"])
    test_dataset = SegmentationCentrioleTest(test_images, test_masks, 
                                             transform=test_transform,
                                             dataset_config=config["dataset"])

    
    dataloaders = {}
    dataloaders['train'] = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], 
                                      shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    dataloaders['val'] = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], 
                                    shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    dataloaders['test'] = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], 
                                     shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    model = UNet(config["architecture"], gamma=config["training"]["gamma"]).to(args.device)
    
    best_model, scores = train(model, dataloaders, args, config)

    torch.save(best_model, f"{args.checkpoint_dir}/best_model.pt")