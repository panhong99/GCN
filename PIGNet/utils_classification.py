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

 
def make_batch_fn(samples, batch_size, feature_shape):
    return make_batch(samples, batch_size, feature_shape)

def visualize_compared_features(compared_features):
    plt.imshow(compared_features, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Compared Features')
    plt.show()


def get_cpu_temperature():
    sensors_output = subprocess.check_output("sensors").decode()
    for line in sensors_output.split("\n"):
        if "Tctl" in line:
            temperature_str = line.split()[1]
            temperature = float(temperature_str[:-3])  # remove "°C" and convert to float
            return temperature

    return None  # in case temperature is not found

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

def zoom_center(image, zoom_factor):
    """
    Zooms into or out of the image around the center by the given zoom_factor.
    Keeps the image size unchanged.
    """
    width, height = image.size

    if zoom_factor > 1:
        # Zoom in
        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)

        # Center coordinates
        center_x, center_y = width // 2, height // 2

        # Calculate the crop box
        left = max(center_x - new_width // 2, 0)
        right = min(center_x + new_width // 2, width)
        top = max(center_y - new_height // 2, 0)
        bottom = min(center_y + new_height // 2, height)

        # Crop the image, then resize back to the original dimensions
        image = image.crop((left, top, right, bottom)).resize((width, height), Image.Resampling.LANCZOS)

    elif zoom_factor < 1:
        # Zoom out
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)

        # Resize the image to the new dimensions
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a new white image and paste the resized image in the center
        new_image = Image.new('RGB', (width, height), (255, 255, 255))
        new_image.paste(resized_image, ((width - new_width) // 2, (height - new_height) // 2))

        image = new_image

    return image
def repeat(image, pattern_repeat_count):
    """
    Repeat the inner region of the image in a grid pattern.
    pattern_repeat_count: Number of repetitions for each dimension (x, y)
    """
    image_size = image.size
    numpy_image = np.array(image)
    original_opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    # Use the entire image as the inner region to repeat
    inner_region = original_opencv_image
    inner_image = Image.fromarray(cv2.cvtColor(inner_region, cv2.COLOR_BGR2RGB))
    inner_image_resize = inner_image.resize((image_size[0], image_size[1]))

    # Create a new empty image of size (image_size[0] * repeat_count, image_size[1] * repeat_count)
    new_image_size = (image_size[0] * pattern_repeat_count, image_size[1] * pattern_repeat_count)
    new_image = Image.new('RGB', new_image_size)

    # Paste the resized inner image in a grid pattern
    for i in range(pattern_repeat_count):
        for j in range(pattern_repeat_count):
            new_image.paste(inner_image_resize, (i * image_size[0], j * image_size[1]))

    # Resize the final repeated image back to the original image size
    final_image = new_image.resize(image_size)

    return final_image
# Define a custom transform class for zoom
class ZoomTransform:
    def __init__(self, zoom_factor):
        self.zoom_factor = zoom_factor

    def __call__(self, image):
        return zoom_center(image, self.zoom_factor)


class RepeatTransform:
    def __init__(self, pattern_repeat_count):
        self.pattern_repeat_count = pattern_repeat_count

    def __call__(self, image):
        return repeat(image, self.pattern_repeat_count)


def resize_pos_embed(posemb , grid_size , new_grid_size , num_extra_tokens = 1):
    #Todo split [CLS] , grid tokens
    posemb_tok , posemb_grid = posemb[: , : num_extra_tokens] , posemb[: , num_extra_tokens :]
    dim = posemb.shape[-1]

    posemb_grid = posemb_grid.reshape(1 , grid_size , grid_size , dim).permute(0 , 3 , 1 , 2) # 1 , dim , H , W
    posemb_grid = F.interpolate(posemb_grid , size = new_grid_size , mode = "bicubic" , align_corners=False)

    posemb_grid = posemb_grid.permute(0 , 2 , 3 , 1).reshape(1 , new_grid_size * new_grid_size , dim)

    #Todo reshape by image size # 32 x 32
    return torch.cat([posemb_tok , posemb_grid] , dim = 1)

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
    print(padded_labels.shape)"""
    if len(samples) < batch_size:

        num_padding = batch_size - len(samples)
        padding_tensor = torch.zeros(((num_padding,)+tuple(inputs[0].shape[:])))
        padded_inputs = torch.cat([torch.stack(inputs), padding_tensor], dim=0)

        padded_labels = torch.cat([torch.stack(labels), torch.zeros((num_padding,)+tuple(labels[0].shape[:]))], dim=0)
        return [padded_inputs, padded_labels]
    else:

        return [torch.stack(inputs), torch.stack(labels)]
