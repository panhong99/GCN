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

warnings.filterwarnings("ignore")

def get_dataset(config):

    if config.dataset == 'imagenet':
            # 데이터셋 경로 및 변환 정의
            image_size=224

            data_dir = '/home/hail/Desktop/HDD/pan/GCN/PIGNet/data/imagenet-100'
            # Set the zoom factor (e.g., 1.2 to zoom in, 0.8 to zoom out)

            if config.mode == "train":
                # Define transformations
                transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),  # Resize to fixed size
                    transforms.ToTensor(),  # Convert image to tensor
                ])

            else:
                if config.infer_params.process_type==None:
                    transform = transforms.Compose([
                        transforms.Resize((image_size, image_size)),  # Resize to fixed size
                        transforms.ToTensor(),  # Convert image to tensor
                    ])
                else:
                    if config.infer_params.process_type == 'zoom':
                        # Define transformations
                        transform = transforms.Compose([
                            transforms.Resize((image_size, image_size)),  # Resize to fixed size
                            utils_classification.ZoomTransform(config.factor),  # Apply the zoom transformation
                            transforms.ToTensor(),  # Convert image to tensor
                        ])
                    elif config.infer_params.process_type =='repeat':

                        # Define transformations
                        transform = transforms.Compose([
                            transforms.Resize((image_size, image_size)),  # Resize to fixed size
                            utils_classification.RepeatTransform(config.factor),  # Apply the repeat transformation
                            transforms.ToTensor(),  # Convert image to tensor
                        ])

                    # TODO Rotate
                    elif config.infer_params.process_type == "rotate":

                        transform = transforms.Compose([
                            transforms.Resize((image_size, image_size)),
                            transforms.Lambda(lambda img: TF.rotate(img, angle=config.factor)),  # (-15 ~ +15) rotate
                            transforms.ToTensor(),
                        ])

            # Load datasets with ImageFolder and apply transformations
            dataset = ImageFolder(root=f'{data_dir}/train', transform=transform)
            valid_dataset = ImageFolder(root=f'{data_dir}/val', transform=transform)

            # dataset = torchvision.datasets.ImageFolder(root=data_dir+'/train', transform=transform)
            #
            # valid_dataset = torchvision.datasets.ImageFolder(root=data_dir+'/val', transform=transform)

            idx2label = []
            cls2label = {}

            import json
            json_file=data_dir+'/Labels.json'
            with open(json_file, "r") as read_file:
                class_idx = json.load(read_file)

                idx2label = list(class_idx.values())

                cla2label = class_idx

                # idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
                # cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}

            dataset.CLASSES = idx2label

    elif config.dataset == 'CIFAR-100':
            image_size = 32

            if config.mode == "train":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 정규화합니다.
                ])

            else:
                
                if config.infer_params.process_type == None:
                    print("original data")
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 정규화합니다.
                    ])

                else:
                    
                    if config.infer_params.process_type == 'zoom':
                        transform = transforms.Compose([
                            utils_classification.ZoomTransform(config.factor),  # Apply the zoom transformation
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 정규화합니다.
                        ])
                    elif config.infer_params.process_type == 'repeat':

                        # Define transformations
                        transform = transforms.Compose([
                            transforms.Resize((image_size, image_size)),  # Resize to fixed size
                            utils_classification.RepeatTransform(config.factor),  # Apply the repeat transformation
                            transforms.ToTensor(),  # Convert image to tensor
                        ])

                    elif config.infer_params.process_type == "rotate":

                        transform = transforms.Compose([
                            transforms.Resize((image_size, image_size)),
                            transforms.Lambda(lambda img: TF.rotate(img, angle=config.factor)),  # (-15 ~ +15) rotate
                            transforms.ToTensor(),
                        ])


            # Load CIFAR-100
            dataset = torchvision.datasets.CIFAR100(root='./data/cifar-100', train=True, download=True, transform=transform)
            valid_dataset = torchvision.datasets.CIFAR100(root='./data/cifar-100', train=False, download=True, transform=transform)

            dataset.CLASSES=sorted(['beaver', 'dolphin', 'otter', 'seal', 'whale',  # aquatic mammals
                            'aquarium' 'fish', 'flatfish', 'ray', 'shark', 'trout',  # fish
                            'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', # flowers
                            'bottles', 'bowls', 'cans', 'cups', 'plates', # food containers
                            'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', # fruit and vegetables
                            'clock', 'computer' 'keyboard', 'lamp', 'telephone', 'television', # household electrical devices
                            'bed', 'chair', 'couch', 'table', 'wardrobe', # household furniture
                            'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', # insects
                            'bear', 'leopard', 'lion', 'tiger', 'wolf', # large carnivores
                            'bridge', 'castle', 'house', 'road', 'skyscraper', # large man-made outdoor things
                            'cloud', 'forest', 'mountain', 'plain', 'sea', # large natural outdoor scenes
                            'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', # large omnivores and herbivores
                            'fox', 'porcupine', 'possum', 'raccoon', 'skunk', # medium-sized mammals
                            'crab', 'lobster', 'snail', 'spider', 'worm', # non-insect invertebrates
                            'baby', 'boy', 'girl', 'man', 'woman', # people
                            'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', # reptiles
                            'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', # small mammals
                            'maple', 'oak', 'palm', 'pine', 'willow', # trees
                            'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', # vehicles 1
                            'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor' # vehicles 2
                            ])

    elif config.dataset == 'CIFAR-10':
            image_size = 32

            if config.mode == "train":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 정규화합니다.
                ])

            else:

                if config.infer_params.process_type == None:
                    print("original data")
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 정규화합니다.
                    ])

                else:
                    if config.infer_params.process_type == 'zoom':
                        transform = transforms.Compose([
                            utils_classification.ZoomTransform(config.factor),  # Apply the zoom transformation
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 정규화합니다.
                        ])
                    elif config.infer_params.process_type == 'repeat':

                        # Define transformations
                        transform = transforms.Compose([
                            transforms.Resize((image_size, image_size)),  # Resize to fixed size
                            utils_classification.RepeatTransform(config.factor),  # Apply the repeat transformation
                            transforms.ToTensor(),  # Convert image to tensor
                        ])
                    # TODO Rotate
                    elif config.infer_params.process_type == "rotate":

                        transform = transforms.Compose([
                            transforms.Resize((image_size, image_size)),
                            transforms.Lambda(lambda img: TF.rotate(img, angle=config.factor)),  # (-15 ~ +15) rotate
                            transforms.ToTensor(),
                        ])

            # CIFAR-10 데이터셋 로드
            dataset = torchvision.datasets.CIFAR10(root='./data/cifar-10/', train=True, download=True,transform=transform)#, transform=transform,target_transform=transform_target)
            valid_dataset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=False, download=True,transform=transform)#, transform=transform,target_transform=transform_target)
            dataset.CLASSES =['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            
    else:
            raise ValueError('Unknown dataset: {}'.format(config.dataset))
    
    
    
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle= True if config.mode == "train" else False            )

    valid_dataset = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False
        )
    
    return dataset , dataset_loader, valid_dataset