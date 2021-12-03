import constants
from tqdm import tqdm
import cv2
import copy
import wandb

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from metrics import metrics



def build_optimizer(config, params):
    filter_fn = filter(lambda p: p.requires_grad, params)
    if config["opt"] == 'adam':
        optimizer = optim.Adam(filter_fn, lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["opt"] == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=config["lr"], momentum=0.95, weight_decay=config["weight_decay"])
    elif config["opt"] == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["opt"] == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=config["lr"], weight_decay=config["weight_decay"])
    return optimizer


def train(model, dataloaders, args, config):
    opt = build_optimizer(config["training"], model.parameters())

    wandb.init(
            project="centriole-segmentation",
            entity="timpostuvan",
            config=config
        )

    best_model = model
    val_max = -np.inf
    for epoch in tqdm(range(config["training"]["epochs"])):
        total_loss = 0
        model.train()
        for (batch, label) in tqdm(dataloaders['train']):
            batch = batch.to(args.device)
            label = label.to(args.device)
            
            opt.zero_grad()
            pred = model(batch)
            loss = model.loss(pred, label)
            total_loss += loss.item()
            loss.backward()
            opt.step()

        scores = test(dataloaders, model, args, config)
        if val_max < scores['val']['acc']:
            val_max = scores['val']['acc']
            best_model = copy.deepcopy(model)


        wandb.log({
                "training loss": total_loss,

                "training accuracy": scores['train']['acc'],
                "validation accuracy": scores['val']['acc'],
                "test accuracy": scores['test']['acc'],

                "training precision": scores['train']['precision'],
                "validation precision": scores['val']['precision'],
                "test precision": scores['test']['precision'],

                "training recall": scores['train']['recall'],
                "validation recall": scores['val']['recall'],
                "test recall": scores['test']['recall'],

                "training average precision": scores['train']['AP_score'],
                "validation average precision": scores['val']['AP_score'],
                "test average precision": scores['test']['AP_score'],

                "training AUC PR": scores['train']['AUC_PRC'],
                "validation AUC PR": scores['val']['AUC_PRC'],
                "test AUC PR": scores['test']['AUC_PRC'],

                "training AUC ROC": scores['train']['AUC_ROC'],
                "validation AUC ROC": scores['val']['AUC_ROC'],
                "test AUC ROC": scores['test']['AUC_ROC'],
            })

        print("Epoch {}:\nTrain: {}\nValidation: {}\nTest: {}\nLoss: {}\n".format(
              epoch + 1, scores['train'], scores['val'], scores['test'], total_loss))


    final_scores = test(dataloaders, best_model, args, config)
    print("FINAL MODEL:\nTrain: {}\nValidation: {}\nTest: {}\n".format(
          final_scores['train'], final_scores['val'], final_scores['test']))
    return best_model, final_scores


def test(dataloaders, model, args, config):
    model.eval()

    scores = {}
    for dataset in dataloaders:
        labels = []
        predictions = []
        for (batch, label) in dataloaders[dataset]:
            batch = batch.to(args.device)
            pred = model(batch)
            predictions.append(pred.flatten().cpu().detach().numpy())
            labels.append(label.flatten().cpu().numpy())

        predictions = torch.tensor(np.concatenate(predictions))
        labels = torch.tensor(np.concatenate(labels))
        print(dataset)
        print(np.histogram(predictions.numpy(), bins=5))
        print(np.histogram(labels.numpy(), bins=5))
        print()
        
        scores[dataset] = metrics(predictions, labels, threshold=config["evaluation"]["threshold"])
    return scores