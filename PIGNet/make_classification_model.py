import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from PIL import Image
from torch.autograd import Variable
from tqdm.auto import tqdm
import pandas as pd
import os
from torchvision import transforms
import math
from model_src import Classification_resnet, PIGNet_GSPonly_classification, swin,PIGNet_classification
# from model_src.cvnets.models.classification import mobilevit_v3
import torch.nn.functional as F
from utils import AverageMeter
from torchvision.datasets import ImageFolder
from functools import partial
import torchvision
import subprocess
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import wandb
from vit_pytorch import ViT
from efficientnet_pytorch import EfficientNet
import warnings
import timm
import torchvision.transforms.functional as TF
import re
import yaml
import copy
import utils_classification as utils_classification
import make_classification_dataset as make_classification_dataset

warnings.filterwarnings("ignore")

def get_model(config, dataset):
    
    if config.backbone in ["resnet50","resnet101"]:

            if config.model_type == "pretrained":
                config.pretrain = True

            else:
                config.pretrain = False
                
            if config.dataset != "imagenet":
                grid_size = 8
                data_stride = 1

            else:
                grid_size = 14
                data_stride = 2
        
            if config.model == "PIGNet_GSPonly_classification":
            
                model = getattr(PIGNet_GSPonly_classification, config.backbone)(
                    pretrained=(config.pretrain),
                    num_classes=len(dataset.CLASSES),
                    num_groups=config.groups,
                    weight_std=config.weight_std,
                    beta=config.beta,
                    embedding_size = config.embedding_size,
                    n_layer = config.n_layer,
                    n_skip_l = config.n_skip_l,
                    data_stride = data_stride,
                    grid_size = grid_size)
                print("model PIGNet_GSPonly_classification")

            elif config.model == "PIGNet_classification":

                model = getattr(PIGNet_classification, config.backbone)(
                    pretrained=(config.pretrain),
                    num_classes=len(dataset.CLASSES),
                    num_groups=config.groups,
                    weight_std=config.weight_std,
                    beta=config.beta,
                    embedding_size = config.embedding_size,
                    n_layer = config.n_layer,
                    n_skip_l = config.n_skip_l,
                    data_stride = data_stride,
                    grid_size = grid_size)
                print("model PIGNet_classification")

            elif config.model == "Resnet":
                print("Classification_resnet model load")
                model = getattr(Classification_resnet, config.backbone)(
                    pretrained=(config.pretrain),
                    num_classes=len(dataset.CLASSES),
                    num_groups=config.groups,
                    weight_std=config.weight_std,
                    beta=config.beta,
                    embedding_size=config.embedding_size,
                    n_layer=config.n_layer,
                    n_skip_l=config.n_skip_l
                    )

            elif config.model == 'vit':
                print(f"classification_vit")

                if config.model_type == "scratch":
                    print("scratch")
                    model = timm.create_model("vit_small_patch16_224" , pretrained = False)
                    
                    if config.dataset != "imagenet":
                        model.patch_embed.img_size = [32 , 32] # 33 for cifar , 224 for imagenet
                        model.patch_embed.proj = nn.Conv2d(3 , 384 , kernel_size = 16 , stride = 16)
                        model.head = nn.Linear(in_features = 384 , out_features = len(dataset.CLASSES))

                        resized_posemb = utils_classification.resize_pos_embed(model.pos_embed , 14 , 2)
                        model.pos_embed = torch.nn.Parameter(resized_posemb)

                elif config.model_type == "pretrained": # pretrained
                    print("pretrained")
                    model = timm.create_model("vit_small_patch16_224" , pretrained = True)

                    if config.dataset != "imagenet":
                        model.patch_embed.img_size = [32 , 32]
                        model.patch_embed.proj = nn.Conv2d(3 , 384 , kernel_size = 16 , stride = 16)
                        model.head = nn.Linear(in_features = 384 , out_features = len(dataset.CLASSES))

                        resized_posemb = utils_classification.resize_pos_embed(model.pos_embed , 14 , 2)
                        model.pos_embed = torch.nn.Parameter(resized_posemb)

            # elif config.model == 'swin':
            #     model = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.DEFAULT)
            # elif config.model == 'mobile':
            #     model = mobilevit_v3()

    else:

            raise ValueError('Unknown backbone: {}'.format(config.backbone))

    size_in_bytes = utils_classification.model_size(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print number of parameters
    print(f"Number of parameters: {num_params / (1000.0 ** 2): .3f} M")

    # num_params_gnn = sum(p.numel() for p in model.pyramid_gnn.parameters() if p.requires_grad)
    # print(f"Number of GNN parameters: {num_params_gnn / (1000.0 ** 2): .3f} M")

    print(f"Entire model size: {size_in_bytes / (1024.0 ** 3): .3f} GB")
    
    return model