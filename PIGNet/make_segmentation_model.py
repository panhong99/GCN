import argparse
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import cv2
from PIL import Image
import torch.nn.functional as F
from scipy.io import loadmat
from torch.autograd import Variable
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import RandomSampler
import pandas as pd
from model_src import PIGNet_GSPonly, ASPP, PIGNet
from model_src.Mask2Former import Mask2Former
from pascal import VOCSegmentation
from cityscapes import Cityscapes
from utils import AverageMeter, inter_and_union
from functools import partial
import subprocess
import wandb
import warnings
import re
import utils_segmentation 
import yaml
import copy

def get_model(config, dataset):
    
    config.scratch = True if config.model_type == "scratch" else False
    
    if config.backbone in ["resnet50","resnet101","resnet152","swin"]:

        if config.model == "PIGNet_GSPonly":
            model = getattr(PIGNet_GSPonly, config.backbone)(
                pretrained=(not config.scratch),
                num_classes=len(dataset.CLASSES),
                num_groups=config.groups,
                weight_std=config.weight_std,
                beta=config.beta,
                embedding_size = config.embedding_size,
                n_layer = config.n_layer,
                n_skip_l = config.n_skip_l)

        elif config.model == "PIGNet":
            model = getattr(PIGNet, config.backbone)(
                pretrained=(not config.scratch),
                num_classes=len(dataset.CLASSES),
                num_groups=config.groups,
                weight_std=config.weight_std,
                beta=config.beta,
                embedding_size = config.embedding_size,
                n_layer = config.n_layer,
                n_skip_l = config.n_skip_l)

        elif config.model == "ASPP":
            model = getattr(ASPP, config.backbone)(
                pretrained=(not config.scratch),
                num_classes=len(dataset.CLASSES),
                num_groups=config.groups,
                weight_std=config.weight_std,
                beta=config.beta,
                embedding_size = config.embedding_size,
                n_layer = config.n_layer,
                n_skip_l = config.n_skip_l)

        elif config.model == "Mask2Former":
            model = Mask2Former(
                backbone = config.backbone,
                num_classes=len(dataset.CLASSES),
                pretrained=(not config.scratch),
                num_groups=config.groups,
                weight_std=config.weight_std,
                beta=config.beta,
                embedding_size = config.embedding_size,
                n_layer = config.n_layer,
                n_skip_l = config.n_skip_l)

    else:
        raise ValueError('Unknown backbone: {}'.format(config.backbone))

    size_in_bytes = utils_segmentation.model_size(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print number of parameters
    print(f"Number of parameters: {num_params / (1000.0 ** 2): .3f} M")

    # num_params_gnn = sum(p.numel() for p in model.pyramid_gnn.parameters() if p.requires_grad)
    # print(f"Number of GNN parameters: {num_params_gnn / (1000.0 ** 2): .3f} M")

    print(f"Entire model size: {size_in_bytes / (1024.0 ** 3): .3f} GB")
    
    return model
    
    