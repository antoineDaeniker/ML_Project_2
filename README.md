# CS-433 Machine Learning - Semantic Segmentation of Centrioles in Human Cells and Assigning Them to Nuclei
This repository contains the code for the second ML project 2 ML4Science, performed in the Gönczy Lab – Cell and Developmental Biology at EPFL.

## Team members
* Antoine Daeniker
* Oliver Becker
* Tim Poštuvan

## Project Layout


## Installation

### Requirements


### Instructions


## Run

### Run U-Net
To run the test of the U-Net and give out predictions use the following command inside the `src` folder:
        
       python 3 run.py --config ../experiments/full_experiment_single_channel.json --num_workers 0

To train the U-Net use the corresponding json under `experiments`.
       
### Run Matching
The notebook `matching.ipynb` shows and explains our matching procedure, from loading the tif image over using StarDist and creating a matching. 
The notebook `matching_bipartite.ipynb` shows and explains bipartite only matching procedure. 

## Data and preparations

### Data
Because of the tif images and their size the data is too big for GitHub. The data can instead be found in: https://drive.google.com/drive/folders/1pQUSt-qwXfVtIBig7JElVzEM0tWha2I0?usp=sharing

### Create one image from all channels
To merge the channel images into one image we created the 'create_channel_images.ipynb' notebook. This goes through the procedure to create the all channel images.
