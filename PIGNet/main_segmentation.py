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
import pickle
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
    
    # 분산 학습 환경변수 확인
    has_world_size = 'WORLD_SIZE' in os.environ
    
    if is_debug or config.mode == "infer" or not has_world_size:
        print("Single GPU mode detected -> skipping distributed setup")
        local_rank = 0
        world_size = 1
        device = f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(config.gpu) if torch.cuda.is_available() else None
        is_distributed = False
    else:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        init_distributed()
        device = f"cuda:{local_rank}"
        is_distributed = True
        print(f"local rank {local_rank}")
        print(f"world size {world_size}")
    
    if local_rank == 0:
        print(f"cuda available : {device}")
    
    is_training = (config.mode == "train")

    if not is_training:
   
        # checkpoint full name
        model_filename = config.infer_params.model_filename

        # model name
        m_name = re.search(fr"(.*?)_{config.backbone}", model_filename)
   
        if m_name:
            config.model = m_name.group(1)
   
        elif "vit" in model_filename:
            config.model = "vit"
   
    else:
        config.model = config.model

    if local_rank == 0:
        print(f"Mode: {config.mode} | Model: {config.model} | Dataset: {config.dataset} | Device: {device}")
    
    loss_data = pd.DataFrame(columns=["train_loss"])
    loss_list = []

    # assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    if config.backbone == "resnet50":
        num=50
    elif config.backbone == "resnet101":
        num=101

    model_fname = f'model_{num}/{config.model_number}/segmentation/{config.dataset}/{config.model_type}/{config.model}_{config.backbone}_{config.model_type}_{config.dataset}_v3.pth'

    if config.mode == "train":
        dataset, valid_dataset = get_dataset(config)
    
    elif config.mode == "infer":
        dataset = get_dataset(config)
        
    model = get_model(config, dataset)

    if config.mode == "train":

        if local_rank == 0:
            print("Training !!! ")
        
        if local_rank == 0:
            wandb.init(project = "gcn_segmentation", name=config.model+"_"+config.backbone+"_"+str(config.model_type)+"_embed"+str(config.embedding_size)+"_nlayer"+str(config.n_layer)+"_"+config.exp+"_"+str(config.dataset),
                config=config.__dict__)

        criterion = nn.CrossEntropyLoss()
        # model = nn.DataParallel(model).to(device)
        model.to(local_rank)
        
        if is_distributed:
            model = DDP(model,
                        device_ids=[local_rank],
                        output_device=local_rank,
                        find_unused_parameters=True)
        
        model.train()

        if config.freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    print("12345")
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

        # Get base model (unwrap DDP if needed)
        base_model = model.module if is_distributed else model

        if config.model != "Mask2Former":
            backbone_params = (
                    list(base_model.conv1.parameters()) +
                    list(base_model.bn1.parameters()) +
                    list(base_model.layer1.parameters()) +
                    list(base_model.layer2.parameters()) +
                    list(base_model.layer3.parameters()) +
                    list(base_model.layer4.parameters()))

        if config.model == "PIGNet_GSPonly" or config.model=="PIGNet":
            last_params = list(base_model.pyramid_gnn.parameters())

        elif config.model == "ASPP":
            last_params = list(base_model.aspp.parameters())

        else: # Masktoformer
            backbone_params = list(base_model.backbone.parameters())
            last_params = list(base_model.transformer_decoder.parameters())

        optimizer = optim.SGD([
            {'params': filter(lambda p: p.requires_grad, backbone_params)},
            {'params': filter(lambda p: p.requires_grad, last_params)}

        ],
            lr=config.base_lr, momentum=0.9, weight_decay=1e-4)

        feature_shape = (2048, 33, 33)

        collate_fn = partial(utils_segmentation.make_batch_fn, batch_size=config.batch_size, feature_shape=feature_shape)

        if is_distributed:
            dataset_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                sampler=DistributedSampler(dataset, 
                                           num_replicas = world_size , 
                                           rank = local_rank),
                pin_memory=True,
                num_workers=config.workers,
                collate_fn=collate_fn
            )
        else:
            dataset_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=config.workers,
                collate_fn=collate_fn
            )

        max_iter = config.epochs * len(dataset_loader)
        losses = AverageMeter()
        start_epoch = 0

        if config.resume:
            if os.path.isfile(config.resume):
                print('=> loading checkpoint {0}'.format(config.resume))
                checkpoint = torch.load(config.resume)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('=> loaded checkpoint {0} (epoch {1})'.format(
                    config.resume, checkpoint['epoch']))
            else:
                print('=> no checkpoint found at {0}'.format(config.resume))

        best_loss = 1e+10
        patience = 0

        train_step = 0

        for epoch in range(start_epoch, config.epochs):

            if is_distributed:
                dataset_loader.sampler.set_epoch(epoch)

            if local_rank == 0:
                print("EPOCHS : ", epoch + 1, " / ", config.epochs)

            loss_sum = 0
            cnt = 0

            if local_rank == 0:
                data_iterator = tqdm(iter(dataset_loader))
            else:
                data_iterator = iter(dataset_loader)
                        
            for inputs, target in data_iterator:
                inputs = inputs.to(local_rank)
                target = target.to(local_rank)
                log = {}
                cur_iter = epoch * len(dataset_loader) + cnt
                lr = config.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
                optimizer.param_groups[0]['lr'] = lr
                # if args.model == "deeplab":
                optimizer.param_groups[1]['lr'] = lr * config.last_mult
                inputs = Variable(inputs.to(device))
                target = Variable(target.to(device)).long()
                # outputs , _, _= model(inputs)
                outputs = model(inputs)

                if config.model == "Mask2Former":
                    outputs = outputs.float()
                else:
                    outputs = outputs[0].float()

                loss = criterion(outputs, target)
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    pdb.set_trace()
                losses.update(loss.item(), config.batch_size)
                loss_sum += loss.item()
                loss.backward()
                #print('batch_loss: {:.5f}'.format(loss))

                log['train/batch/loss'] = loss.to('cpu')

                #log['train/temp'] = get_cpu_temperature()

                cnt += 1
                train_step += 1

                optimizer.step()
                optimizer.zero_grad()

                # if train_step % 30 == 0:
                #     time.sleep(15)

            loss_avg = loss_sum / len(dataset_loader)
            log['train/epoch/loss'] = loss_avg

            if local_rank == 0:
                wandb.log(log)

            loss_list.append(loss_avg)
            print('epoch: {0}\t'
                  'lr: {1:.6f}\t'
                  'loss: {2:.4f}'.format(
                epoch + 1, lr, loss_avg))

            if best_loss > loss_avg:
                best_loss = loss_avg
                patience = 0

                # DDP일 때와 단일 GPU일 때 다르게 처리
                state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                
                os.makedirs(os.path.dirname(model_fname), exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                }, model_fname)
                print("model save complete!!!")
            else:
                patience += 1
                print("patience : ", patience)

                if patience > 50:
                    break
            # time.sleep(150)
            print("Evaluating !!! ")

            inter_meter = AverageMeter()
            union_meter = AverageMeter()
            
            if is_distributed:
                valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
                valid_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=config.batch_size, 
                    shuffle=False,
                    sampler=valid_sampler,
                    pin_memory=True,
                    num_workers=config.workers,
                    collate_fn=collate_fn 
                )
            else:
                valid_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=config.batch_size, 
                    shuffle=False,
                    pin_memory=True,
                    num_workers=config.workers,
                    collate_fn=collate_fn 
                )

            with torch.no_grad():
                model.eval()
                losses_test = 0.0
                log = {}
                
                if local_rank == 0:
                    eval_iterator = tqdm(valid_loader)
                else:
                    eval_iterator = valid_loader

                for inputs, target in eval_iterator:
                    
                    # inputs, target = valid_dataset[i]

                    inputs = inputs.to(local_rank)
                    target = target.to(local_rank).long()

                    # outputs = model(inputs.unsqueeze(0))
                    outputs = model(inputs)
                    
                    if config.model == "Mask2Former":
                        outputs = outputs.float()
                    else:
                        outputs = outputs[0].float()

                    _, pred = torch.max(outputs, 1)
    
                    # pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
                    # mask = target.cpu().numpy().astype(np.uint8)
    
                    pred = pred.data.cpu().numpy().astype(np.uint8)
                    mask = target.cpu().numpy().astype(np.uint8)

                    # inter, union = inter_and_union(pred, mask, len(valid_dataset.CLASSES))
                    # inter_meter.update(inter)
                    # union_meter.update(union)
                    
                    for p, m in zip(pred, mask):
                        inter, union = inter_and_union(p, m, len(valid_dataset.CLASSES))
                        inter_meter.update(inter)
                        union_meter.update(union)



                    # calculate loss
                    loss = criterion(outputs, target)
                    # add to losses_test
                    losses_test += loss.item() * inputs.size(0)

                    # log['test/batch/loss'] = loss.to('cpu')
                    # log['test/batch/iou'] = (inter.sum() / (union.sum() + 1e-10))

                train_step += 1

                if is_distributed:
                    inter_sum = torch.tensor(inter_meter.sum, device=local_rank)
                    union_sum = torch.tensor(union_meter.sum, device=local_rank)

                    dist.all_reduce(inter_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(union_sum, op=dist.ReduceOp.SUM)

                    # 합산된 결과를 CPU로 가져와 NumPy로 변환
                    final_inter = inter_sum.cpu().numpy()
                    final_union = union_sum.cpu().numpy()
                else:
                    final_inter = inter_meter.sum
                    final_union = union_meter.sum

                # iou = inter_meter.sum / (union_meter.sum + 1e-10)
                # miou = iou.mean()
                
                iou = final_inter / (final_union + 1e-10)
                miou = iou.mean()
                
                if is_distributed:
                    total_valid_samples = len(valid_dataset) * world_size
                    total_losses_test_tensor = torch.tensor(losses_test, device=local_rank)
                    dist.all_reduce(total_losses_test_tensor, op=dist.ReduceOp.SUM)
                    total_losses_test = total_losses_test_tensor.item()
                    total_samples = len(valid_dataset) * world_size
                else:
                    total_losses_test = losses_test
                    total_samples = len(valid_dataset)
                
                if local_rank == 0:
                    for i, val in enumerate(iou):
                        print('IoU {0}: {1:.2f}'.format(valid_dataset.CLASSES[i], val * 100))
                    print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))
                    
                    avg_valid_loss = total_losses_test / total_samples 
                    log['test/epoch/loss'] = avg_valid_loss
                    log['test/epoch/iou'] = miou.item()
        
            if local_rank == 0:
                wandb.log(log)
                model.train()
        
        wandb.finish()
        
    else:        
        print("Evaluating !!! ")
        torch.cuda.set_device(config.gpu)
        model = model.to(device)
        model.eval()

        checkpoint = torch.load(f'/home/hail/pan/GCN/PIGNet/model_{num}/{config.model_number}/segmentation/{config.dataset}/{config.model_type}/{model_filename}'
                                , map_location = device)

        state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
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

        # 이미지 데이터를 메모리에 저장할 리스트
        pred_img = []
        iou_list = []
        img_name = []

        for i in tqdm(range(len(dataset))):

            inputs, target, _, color_target, _, _ = dataset[i]
            if inputs==None:
                continue

            inputs = Variable(inputs.to(device))

            if config.model == "Mask2Former":
                outputs = model(inputs.unsqueeze(0))

            else:
                outputs, _, _ = model(inputs.unsqueeze(0))

            _, pred = torch.max(outputs, 1)
            pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
            mask = target.numpy().astype(np.uint8)
            
            # search padding location
            # pred = pred[: pred.shape[0] - H, : pred.shape[1] - W]
            # mask = mask[: mask.shape[0] - H, : mask.shape[1] - W]

            inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
            iou_score = inter.sum() / union.sum()
            
            if config.dataset == "pascal":
                thredhold = 0.1
            elif config.dataset == "cityscape":
                thredhold = 0.1
            
            if iou_score > thredhold:
                # 각각 별도의 리스트에 저장
                pred_img.append(pred)
                iou_list.append(iou_score)
                img_name.append(dataset.images[i].split('/')[-1])
                        
            inter_meter.update(inter)
            union_meter.update(union)
                                    
        print('eval: {0}/{1}'.format(i + 1, len(dataset)))

        # 데이터를 하나의 딕셔너리로 통합 (모든 값을 CPU로 이동 및 stack)
        output_data = {
            'pred_img': np.stack([np.asarray(x, dtype=np.uint8) for x in pred_img]),
            'iou': np.array([float(x.cpu()) if isinstance(x, torch.Tensor) else float(x) for x in iou_list]),
            'img_name': np.array(img_name, dtype=object)
        }
        
        # Pickle 파일로 저장 (단일 파일)
        base_path = f'/home/hail/pan/GCN/PIGNet/infer_output'
        os.makedirs(base_path, exist_ok=True)
        
        if config.factor == np.sqrt(0.1):
            config.factor = 0.3
        elif config.factor == np.sqrt(0.5):
            config.factor = 0.7
        elif config.factor == np.sqrt(2.75):
            config.factor = 1.75
        
        pkl_path = f'{base_path}/{config.dataset}_{config.model}_{config.infer_params.process_type}_{config.factor}_number_{config.model_number}.pkl'
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(output_data, f)

        iou = inter_meter.sum / (union_meter.sum + 1e-10)

        for i, val in enumerate(iou):
            print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
        print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))
        return iou.mean() * 100

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Load configuration from config.yaml")
    parser.add_argument("--config", type = str, default = "/home/hail/pan/GCN/PIGNet/config_segmentation.yaml", help = "path to config.yaml")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error config.yaml not found at {args.config}")
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

    if config.mode == "train":

        print("-- Starting Traning Mode --")
        
        main(config)
        
    elif config.mode == "infer":
        print("-- Starting Infer Mode --")
        
        if config.backbone == "resnet50":
            num=50
        elif config.backbone == "resnet101":
            num=101
                    
        path = f"/home/hail/pan/GCN/PIGNet/model_{num}/{config.model_number}/segmentation/{config.dataset}/{config.model_type}"
                
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
    
                    accuracy = main(iter_config)

                    if accuracy is not None:
                        output_dict[name][process_key].append(accuracy)
                        
        print("\n--- Inference Results Summary ---")

        records = []
        for model_name, result_dict in output_dict.items():
            for task, values in result_dict.items():
                factors = process_dict.get(task, [])
                for i, val in enumerate(values):
                    records.append({
                        "model": model_name,
                        "task": task,
                        "factor": factors[i] if i < len(factors) else "N/A",
                        "accuracy": val
                    })
        df_long = pd.DataFrame(records)
        
        df_wide = df_long.pivot_table(index=['model', 'task'], 
                                      columns='factor', 
                                      values='accuracy').reset_index()
        
        df_wide.rename_axis(columns=None, inplace=True)

        output_filename = f"output_{num}_{config.model_number}_{config.model_type}_{config.dataset}.csv"
        df_wide.to_csv(output_filename, index=False)
        
        print(f"\n[SUCCESS] Reshaped results saved to '{output_filename}'")
        print(df_wide)

    else:
        print(f"[ERROR] Unknown mode: '{config.mode}'. Please set mode to 'train' or 'infer' in '{args.config}'")