import os
import sys
import glob
import json
import argparse
from tqdm import tqdm
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import constants
import utils
import preprocessing

from datasets import SegmentationCentrioleTrain, SegmentationCentrioleTest
from model import UNet
from train import train



def arg_parse():
    """
    Add element to parser from the cmd line

    Returns:
        parser: the parser with all parse argument from the cmd line
    """
    parser = argparse.ArgumentParser(description='Centriole segmentation arguments.')

    parser.add_argument("--config", type=str, required=True, 
                        help="The config file that contains all hyperparameters")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='CPU / GPU device.')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='CPU / GPU device.')
    parser.add_argument('--dataset_dir', type=str, default="../dataset",
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

    assert config["dataset"]["channels"] in [1, 3], "Only 1 or 3 channels are supported!"

    image_format = "tif"
    channels_dir = "single-channel-images"
    if(config["dataset"]["channels"] == 3):
        channels_dir = "all-channel-images"

    image_path = f"{args.dataset_dir}/{channels_dir}/{image_format}"
    mask_path = f"{args.dataset_dir}/{channels_dir}/masks"
    
    # relative paths to images
    all_masks = []
    for mask in glob.glob(f"{mask_path}/*.png"):
        all_masks.append(mask)
        print(mask)
    
    all_masks.sort()

    # train/val/test split: 80/10/10
    train_masks, val_masks = train_test_split(all_masks, test_size=0.2)
    val_masks, test_masks = train_test_split(val_masks, test_size=0.5)

    train_images = [f'{image_path}/{mask.split("/")[-1][:-9]}.{image_format}' for mask in train_masks] 
    val_images = [f'{image_path}/{mask.split("/")[-1][:-9]}.{image_format}' for mask in val_masks] 
    test_images = [f'{image_path}/{mask.split("/")[-1][:-9]}.{image_format}' for mask in test_masks] 
   
    print(f"Training set: {len(train_images)} instances.")
    print(f"Validation set: {len(val_images)} instances.")
    print(f"Test set: {len(test_images)} instances.")


    data_augmentation = {}
    if(config["dataset"]["data_augmentation"]):
        data_augmentation["both"] = transforms.Compose([
                                                        transforms.RandomHorizontalFlip(p=0.2),
                                                        transforms.RandomVerticalFlip(p=0.2)
                                                    ])
        data_augmentation["image"] = transforms.Compose([preprocessing.RandomGaussianNoise(mean=0, std=1)])
    else:
        data_augmentation["both"] = None
        data_augmentation["image"] = None


    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])


    train_dataset = SegmentationCentrioleTrain(train_images, train_masks, 
                                               transform=train_transform,
                                               dataset_config=config["dataset"],
                                               data_augmentation=data_augmentation)
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
    
    # load the trained weights if they exist
    checkpoint_path = f'{args.checkpoint_dir}/{args.config.split("/")[-1][:-5]}'
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    try:
        print(f"{checkpoint_path}/best_model.pt")
        best_model = torch.load(f"{checkpoint_path}/best_model.pt", map_location=torch.device(args.device))
        best_model.eval()
        print("Trained weights successfully loaded.")
    except:
        print("There are no trained weights. Initializing training.")
        best_model, scores = train(model, dataloaders, args, config)
        torch.save(best_model, f"{checkpoint_path}/best_model.pt")
        best_model.eval()

    # tile predicted masks for visualization purposes
    predictions = []
    for (batch, label) in tqdm(dataloaders['test']):
        batch = batch.to(args.device)

        pred = F.softmax(best_model(batch), dim=1)
        for i in range(pred.shape[0]):
            predictions.append(pred[i, 1].cpu().detach().unsqueeze(dim=0))

    # resize to original size
    resize_transform = transforms.Resize(size=(constants.FULL_IMAGE_WIDTH,
                                               constants.FULL_IMAGE_HEIGHT),
										interpolation=InterpolationMode.NEAREST)


    predictions_path = f'{args.dataset_dir}/{channels_dir}/predictions/{args.config.split("/")[-1][:-5]}'
    Path(predictions_path).mkdir(parents=True, exist_ok=True)
    test_predictions = [f'{predictions_path}/{image.split("/")[-1]}' for image in test_masks]       
    for im in range(len(test_images)):
        vertical_concat = []
        for i in range(test_dataset.height_crops):
            horizontal_concat = []
            for j in range(test_dataset.width_crops):
                idx = (im * test_dataset.crops_per_image 
                     + i * test_dataset.width_crops
                     + j)
                horizontal_concat.append(predictions[idx])
            
            horizontal_concat = torch.cat(horizontal_concat, dim=2)
            vertical_concat.append(horizontal_concat)

        vertical_concat = torch.cat(vertical_concat, dim=1)

        prediction_mask = resize_transform(vertical_concat).permute(1, 2, 0).numpy()
        cv2.imwrite(test_predictions[im], prediction_mask * 255)