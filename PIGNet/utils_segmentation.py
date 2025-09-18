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

def tensor_to_image(tensor_image):
    
    if tensor_image.shape[0] <= 3:
        image = tensor_image.permute(1,2,0).detach().cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    else: # shape[0] > 3
        image = tensor_image.detach().cpu().numpy()
        image = image.astype(np.uint8)
        image = Image.fromarray(image)    

    return image

def make_batch_fn(samples, batch_size, feature_shape):
    return make_batch(samples, batch_size, feature_shape)

def find_contours(mask):
    mask_array = np.array(mask)
    _, binary_mask = cv2.threshold(mask_array, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_cpu_temperature():
    sensors_output = subprocess.check_output("sensors").decode()
    for line in sensors_output.split("\n"):
        if "Tctl" in line:
            temperature_str = line.split()[1]
            temperature = float(temperature_str[:-3])  # remove "°C" and convert to float
            return temperature

    return None  # in case temperature is not found

# 유클리드 거리에 따라 일정 거리에 있는 좌표들을 찾는 함수
def get_coords_by_distance(center_x, center_y, distance, feature_map_size_h, feature_map_size_w):
    coords = []
    for i in range(feature_map_size_h):
        for j in range(feature_map_size_w):
            dist = math.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            if abs(dist - distance) < 0.5:  # 거리 차이가 0.5 이내면 같은 거리로 간주
                coords.append((i, j))
    return coords


# 코사인 유사도 계산 함수
def calculate_cosine_similarity(coords, center_vector, tensor):
    cos_sims = []
    for (x, y) in coords:
        vector = tensor[0, :, x, y]  # 해당 좌표의 512차원 벡터
        cos_sim = F.cosine_similarity(center_vector, vector, dim=0)  # 코사인 유사도 계산
        cos_sims.append(cos_sim.item())
    return cos_sims

def model_size(model):
    total_size = 0
    for param in model.parameters():
        num_elements = torch.prod(torch.tensor(param.size())).item()
        num_bytes = num_elements * param.element_size()
        total_size += num_bytes
    return total_size

def make_batch(samples, batch_size, feature_shape):
    inputs = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]

    """
    print(inputs[0].shape)
    print(labels[0].shape)
    padding_tensor = torch.zeros(((2,) + tuple(inputs[0].shape[:])))
    print(padding_tensor.shape,torch.stack(inputs).shape)
    padded_inputs = torch.cat([torch.stack(inputs), padding_tensor], dim=0)
    print(padded_inputs.shape)
    padded_labels = torch.cat([torch.stack(labels), torch.zeros((2,)+tuple(labels[0].shape[:]))], dim=0)
    print(padded_labels.shape)
    """

    if len(samples) < batch_size:

        num_padding = batch_size - len(samples)
        padding_tensor = torch.zeros(((num_padding,)+tuple(inputs[0].shape[:])))
        padded_inputs = torch.cat([torch.stack(inputs), padding_tensor], dim=0)

        padded_labels = torch.cat([torch.stack(labels), torch.zeros((num_padding,)+tuple(labels[0].shape[:]))], dim=0)
        return [padded_inputs, padded_labels]
    else:

        return [torch.stack(inputs), torch.stack(labels)]