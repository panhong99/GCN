import argparse
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import cv2
import matplotlib.pyplot as plt
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
from make_segmentation_dataset import get_dataset
from make_segmentation_model import get_model
import random
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys

warnings.filterwarnings("ignore")

def init_distributed():  
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)                       
    dist.init_process_group("nccl",                   
                            rank=local_rank,               
                            world_size=world_size)

def set_seed(seed_value=42):

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.deterministic = True
        
        torch.backends.cudnn.benchmark = False
        
cityscapes_colormap = {
    0: (128, 64, 128),   # road
    1: (244, 35, 232),   # sidewalk
    2: (70, 70, 70),     # building
    3: (102, 102, 156),  # wall
    4: (190, 153, 153),  # fence
    5: (153, 153, 153),  # pole
    6: (250, 170, 30),   # traffic light
    7: (220, 220, 0),    # traffic sign
    8: (107, 142, 35),   # vegetation
    9: (152, 251, 152),  # terrain
    10: (70, 130, 180),  # sky
    11: (220, 20, 60),   # person
    12: (255, 0, 0),     # rider
    13: (0, 0, 142),     # car
    14: (0, 0, 70),      # truck
    15: (0, 60, 100),    # bus
    16: (0, 80, 100),    # on rails
    17: (0, 0, 230),     # motorcycle
    18: (119, 11, 32),   # bicycle
    255: (0, 0, 0)       # unlabeled / void
}

palette = np.zeros((256, 3), dtype=np.uint8)

for train_id, color in cityscapes_colormap.items():
    if train_id < 256:
        palette[train_id] = color

cityscapes_cmap = palette.flatten().tolist()

def main(config): 
    # 디버깅 모드 감지
    is_debug = (hasattr(sys, 'gettrace') and sys.gettrace()) or os.getenv('DEBUG', '') == '1'

    local_rank = 0  # Always use local_rank=0 for this script
    device = f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(config.gpu) if torch.cuda.is_available() else None

    print(f"cuda available : {device}")
    
    model_filename = config.infer_params.model_filename

    # model name
    m_name = re.search(fr"(.*?)_{config.backbone}", model_filename)

    if m_name:
        config.model = m_name.group(1)

    elif "vit" in model_filename:
        config.model = "vit"

    print(f"Model: {config.model} | Dataset: {config.dataset} | Device: {device}")
    
    # assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(config.gpu)

    model_fname = f'model/{config.model_number}/segmentation/{config.dataset}/{config.model_type}/{config.model}_{config.backbone}_{config.model_type}_{config.dataset}_v3.pth'
    
    dataset = get_dataset(config)
        
    model = get_model(config, dataset)
    model = model.to(device)
    model.eval()

    checkpoint = torch.load(f'/home/hail/pan/GCN/PIGNet/model/{config.model_number}/segmentation/{config.dataset}/{config.model_type}/{model_filename}'
                            , map_location = device)

    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    print(model_fname)
    model.load_state_dict(state_dict)

    if config.dataset == "pascal":
        cmap = loadmat('/home/hail/pan/GCN/PIGNet/data/pascal_seg_colormap.mat')['colormap']
        cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

    elif config.dataset == "cityscape":
        cmap = cityscapes_cmap

    inter_meter = AverageMeter()
    union_meter = AverageMeter()

    feature_shape = (2048, 33, 33)

    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False,
        pin_memory=True, num_workers=config.workers,
        collate_fn=lambda samples: utils_segmentation.make_batch(samples, config.batch_size, feature_shape))

    output_folder = os.path.join("MI_dataset", config.dataset, config.infer_params.process_type, config.model, str(config.factor))
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving to folder: {output_folder}")

    i = 0
    while i < len(dataset):
        inputs, target, gt_image, color_target, H, W = dataset[i]
        if inputs is not None:
            break
        i += 1
    else:
        print("No valid input found.")
        return

    inputs = Variable(inputs.to(device))

    if config.model == "Mask2Former":
        outputs = model(inputs.unsqueeze(0))
        layers_output = None  # Mask2Former는 layer output 없음
    else:
        outputs, layers_output, backbone_layers_output = model(inputs.unsqueeze(0))

    _, pred = torch.max(outputs, 1)
    pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
    mask = target.numpy().astype(np.uint8)
    
    # Padding 제거
    pred = pred[: pred.shape[0] - H, : pred.shape[1] - W]
    mask = mask[: mask.shape[0] - H, : mask.shape[1] - W]

    if config.dataset == "cityscape":
        pad_location = (mask == 255)
        pred[pad_location] = 255
            
    imname = dataset.masks[i].split('/')[-1]
    mask_pred = Image.fromarray(pred)
    mask_pred.putpalette(cmap)

    # 저장: 하나의 폴더에 모든 파일
    mask_pred.save(os.path.join(output_folder, "pred_seg.png"))
    gt_image.save(os.path.join(output_folder, "GT_input.png"))
    color_target.save(os.path.join(output_folder, "GT_seg.png"))
    
    if layers_output is not None:
        torch.save(layers_output, os.path.join(output_folder, "layers_activity.pth"))
    
    print(f"Saved: pred_seg.png, GT_input.png, GT_seg.png, layers_activity.pth in {output_folder}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load configuration from config.yaml")
    parser.add_argument("--config", type = str, default = "/home/hail/pan/GCN/PIGNet/config_segmentation.yaml", help = "path to config.yaml")
    cli_args = parser.parse_args()

    try:
        with open(cli_args.config, "r") as f:
            config_dict = yaml.safe_load(f)

    except FileNotFoundError:
        print(f"Error config.yaml not found at {cli_args.config}")
        exit()

    except Exception as e:
        print(f"Error parsing YAML file: {e}")
        exit()
        
    def dict_to_namespace(d):
        namespace = argparse.Namespace()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(namespace, k, dict_to_namespace(v))
            else:
                setattr(namespace, k, v)
        return namespace
    
    config = dict_to_namespace(config_dict)

    print("-- Infer Mode --")
    
    path = f"/home/hail/pan/GCN/PIGNet/model/{config.model_number}/segmentation/{config.dataset}/{config.model_type}"
            
    try:
        model_list = sorted(os.listdir(path))
        print(f"[INFO] Found {len(model_list)} models in '{path}'")

    except FileNotFoundError:
        print(f"[ERROR] Model directory not found at '{path}'")
        exit()
    
    zoom_factor = [0.1, np.sqrt(0.1), 0.5, np.sqrt(0.5), 1, 1.5, np.sqrt(2.75), 2] # zoom in, out value 양수면 줌 음수면 줌아웃

    overlap_percentage = [0, 0.1 , 0.2 , 0.3 , 0.5] #겹치는 비율 0~1 사이 값으로 0.8 이상이면 shape 이 안맞음

    pattern_repeat_count = [1, 3, 6, 9, 12] # 반복 횟수 2이면 2*2

    output_dict = {model_name : {"zoom" : [] , "overlap" : [] , "repeat" : []} for model_name in model_list}

    process_dict = {
        "zoom" : zoom_factor , 
        "overlap" : overlap_percentage ,
        "repeat" : pattern_repeat_count
    }
    
    # zoom_factor = [1] # zoom in, out value 양수면 줌 음수면 줌아웃

    # # overlap_percentage = [0.3] #겹치는 비율 0~1 사이 값으로 0.8 이상이면 shape 이 안맞음

    # # pattern_repeat_count = [1, 3, 6, 9, 12] # 반복 횟수 2이면 2*2

    # output_dict = {model_name : {"zoom" : []} for model_name in model_list}

    # process_dict = {
    #     "zoom" : zoom_factor , 
    # }
    
    for name in model_list:
        for process_key , factor_list in process_dict.items():
            for factor_value in factor_list:
                
                iter_config = copy.deepcopy(config)
                if "Mask2Former" in name:
                    iter_config.crop_size = 512
                iter_config.infer_params.model_filename = name
                iter_config.infer_params.process_type = process_key
                iter_config.factor = factor_value

                print("-" * 60)
                print(f"Testing model: {name} | Process: {process_key} | Factor: {factor_value}")
                print("\n")

                main(iter_config)

