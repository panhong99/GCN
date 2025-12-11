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
import utils_segmentation as utils_segmentation
import yaml
import copy

def get_dataset(config):

    config.train = True if config.mode == "train" else False

    if config.dataset == 'pascal':

        if config.mode == "train":

            print("train dataset")
            dataset = VOCSegmentation('/home/hail/pan/GCN/PIGNet/data/VOCdevkit',
                                      train=config.train, crop_size=config.crop_size)
            valid_dataset = VOCSegmentation('/home/hail/pan/GCN/PIGNet/data/VOCdevkit',
                                            train=not (config.train), crop_size=config.crop_size)
        else:

            if config.infer_params.process_type != None:
                print(config.infer_params.process_type)
                dataset = VOCSegmentation('/home/hail/pan/GCN/PIGNet/data/VOCdevkit',
                                                train=config.train, crop_size=config.crop_size,
                                                process=config.infer_params.process_type, process_value=config.factor,
                                                overlap_percentage=config.factor,
                                                pattern_repeat_count=config.factor)
            else:
                dataset = VOCSegmentation('/home/hail/pan/GCN/PIGNet/data/VOCdevkit',
                                                train=config.train, crop_size=config.crop_size,
                                                process=None, process_value=config.factor,
                                                overlap_percentage=config.factor,
                                                pattern_repeat_count=config.factor)

    elif config.dataset == 'cityscape':

        if config.train:
            print("train dataset cityscape")

            dataset = Cityscapes('/home/hail/pan/GCN/PIGNet/data/cityscape',
                                 train=config.train, crop_size=config.crop_size)

            valid_dataset = Cityscapes('/home/hail/pan/GCN/PIGNet/data/cityscape',
                                 train=not (config.train), crop_size=config.crop_size)

        else: # val
            if config.infer_params.process_type != None:
                print(config.infer_params.process_type)
                dataset = Cityscapes('/home/hail/pan/GCN/PIGNet/data/cityscape',
                                          train=config.train, crop_size=config.crop_size,
                                          process=config.infer_params.process_type, process_value=config.factor,
                                          overlap_percentage=config.factor,
                                          pattern_repeat_count=config.factor)
            else:
                dataset = Cityscapes('/home/hail/pan/GCN/PIGNet/data/cityscape',
                                          train=config.train, crop_size=config.crop_size,
                                          process=None, process_value=config.factor,
                                          overlap_percentage=config.factor,
                                          pattern_repeat_count=config.factor)

    else:
        raise ValueError('Unknown dataset: {}'.format(config.dataset))
    
    if config.train:
        return dataset, valid_dataset
    else:
        return dataset