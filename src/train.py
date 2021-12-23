from tqdm import tqdm
import copy
import wandb

import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix

import constants
import utils
from metrics import metrics, metrics_from_confusion_matrix



def build_optimizer(config, params):
    """
    Build the optimizer

    Args:
        config : choose which optimizer, lerning rate, weight_decay we want
        params : filter parameter

    Returns:
        optimizer: optimizer acording to config and params args
    """
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
    """
    Training procedure of the model

    Args:
        model : the model we choose
        dataloaders : represent the dataset
        args : args to use wandb
        config : procedure configurations

    Returns:
        best_model : the best model during the training procedure
        final_scores : all results from the testing procedure
    """
    opt = build_optimizer(config["training"], model.parameters())

    # scheduler
    scheduler_gamma = 0.9
    lambda1 = lambda epoch: (1 - epoch/config["training"]["epochs"])**scheduler_gamma 
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)

    if(args.use_wandb):
        wandb.init(
                project="centriole-segmentation",
                entity="centriole-segmentation",
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
        if val_max < scores['val']['f1_score']:
            val_max = scores['val']['f1_score']
            best_model = copy.deepcopy(model)

        if(scheduler is not None):
            scheduler.step()

        if(args.use_wandb):
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

                    "training f1 score": scores['train']['f1_score'],
                    "validation f1 score": scores['val']['f1_score'],
                    "test f1 score": scores['test']['f1_score'],

                    "training IoU": scores['train']['IoU'],
                    "validation IoU": scores['val']['IoU'],
                    "test IoU": scores['test']['IoU']
                })

        print("Epoch {}:\nTrain: {}\nValidation: {}\nTest: {}\nLoss: {}\n".format(
              epoch + 1, scores['train'], scores['val'], scores['test'], total_loss))


    final_scores = test(dataloaders, best_model, args, config)
    print("FINAL MODEL:\nTrain: {}\nValidation: {}\nTest: {}\n".format(
          final_scores['train'], final_scores['val'], final_scores['test']))
    return best_model, final_scores


def test(dataloaders, model, args, config):
    """
    Testing procedure

    Args:
        dataloaders : represent the dataset
        model : the model we choose
        args : args to use wandb
        config : procedure configurations

    Returns:
        scores : test results using training prediction
    """
    model.eval()

    scores = {}
    for dataset in dataloaders:
        total_confusion_matrix = np.zeros((constants.NUM_CLASSES, constants.NUM_CLASSES))
        for (batch, label) in tqdm(dataloaders[dataset]):
            batch = batch.to(args.device)

            pred = F.softmax(model(batch), dim=1)
            argmax_pred = torch.argmax(pred, dim=1).unsqueeze(1)            
            thresholded_pred = (pred[:, 1, :, :] >= config["evaluation"]["threshold"])

            cur_confusion_matrix = confusion_matrix(label.flatten().cpu().detach().numpy(),
                                                    thresholded_pred.flatten().cpu().detach().numpy(),
                                                    labels=np.arange(constants.NUM_CLASSES))
            total_confusion_matrix += cur_confusion_matrix

            """
            for i in range(batch.shape[0]):
                stacked = torch.cat((batch[i].detach(), argmax_pred[i].detach()), dim=0)
                utils.plot_image_mask(stacked.permute(1, 2, 0), label[i].detach().permute(1, 2, 0))
            """

        scores[dataset] = metrics_from_confusion_matrix(total_confusion_matrix)

        print(dataset)
        print(total_confusion_matrix)
    return scores