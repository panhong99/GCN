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
from make_segmentation_dataset import get_dataset
from make_segmentation_model import get_model

warnings.filterwarnings("ignore")

def main(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"    
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

    print(f"Mode: {config.mode} | Model: {config.model} | Dataset: {config.dataset} | Device: {device}")
    
    loss_data = pd.DataFrame(columns=["train_loss"])
    loss_list = []

    # assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    model_fname = f'model/{config.model_number}/segmentation/{config.dataset}/{config.model_type}/segmentation{config.model}_{config.backbone}_{config.model_type}_{config.dataset}_v3.pth'

    if config.mode == "train":
        dataset, valid_dataset = get_dataset(config)
    
    elif config.mode == "infer":
        dataset = get_dataset(config)
        
    model = get_model(config, dataset)

    if config.mode == "train":
        print("Training !!! ")

        criterion = nn.CrossEntropyLoss(ignore_index=255)
        model = nn.DataParallel(model).to(device)
        model.train()

        if config.freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    print("12345")
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

        if config.model != "Mask2Former":
            backbone_params = (
                    list(model.module.conv1.parameters()) +
                    list(model.module.bn1.parameters()) +
                    list(model.module.layer1.parameters()) +
                    list(model.module.layer2.parameters()) +
                    list(model.module.layer3.parameters()) +
                    list(model.module.layer4.parameters()))

        if config.model == "PIGNet_GSPonly" or config.model=="PIGNet":
            last_params = list(model.module.pyramid_gnn.parameters())

        elif config.model == "ASPP":
            last_params = list(model.module.aspp.parameters())

        else: # Masktoformer
            backbone_params = list(model.module.backbone.parameters())
            last_params = list(model.module.transformer_decoder.parameters())

        optimizer = optim.SGD([
            {'params': filter(lambda p: p.requires_grad, backbone_params)},
            {'params': filter(lambda p: p.requires_grad, last_params)}

        ],
            lr=config.base_lr, momentum=0.9, weight_decay=1e-4)

        feature_shape = (2048, 33, 33)

        collate_fn = partial(utils_segmentation.make_batch_fn, batch_size=config.batch_size, feature_shape=feature_shape)

        dataset_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.train,
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
            print("EPOCHS : ", epoch + 1, " / ", config.epochs)

            loss_sum = 0
            cnt = 0

            for inputs, target in tqdm(iter(dataset_loader)):
                log = {}
                cur_iter = epoch * len(dataset_loader) + cnt
                lr = config.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
                optimizer.param_groups[0]['lr'] = lr
                # if args.model == "deeplab":
                optimizer.param_groups[1]['lr'] = lr * config.last_mult
                inputs = Variable(inputs.to(device))
                target = Variable(target.to(device)).long()
                outputs , _ = model(inputs)
                outputs = outputs.float()
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

            wandb.log(log)

            loss_list.append(loss_avg)
            print('epoch: {0}\t'
                  'lr: {1:.6f}\t'
                  'loss: {2:.4f}'.format(
                epoch + 1, lr, loss_avg))

            if best_loss > loss_avg:
                best_loss = loss_avg
                patience = 0

                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
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

            with torch.no_grad():
                model.eval()
                losses_test = 0.0
                log = {}
                for i in tqdm(range(len(valid_dataset))):
                    inputs, target = valid_dataset[i]
                    inputs = Variable(inputs.to(device))
                    target = Variable(target.to(device)).long()
                    outputs , _ = model(inputs.unsqueeze(0))
                    outputs = outputs.float()
                    _, pred = torch.max(outputs, 1)
                    pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
                    mask = target.cpu().numpy().astype(np.uint8)

                    inter, union = inter_and_union(pred, mask, len(valid_dataset.CLASSES))
                    inter_meter.update(inter)
                    union_meter.update(union)

                    # calculate loss
                    loss = criterion(outputs, target.unsqueeze(0))
                    # add to losses_test
                    losses_test += loss.item()

                    # log['test/batch/loss'] = loss.to('cpu')
                    # log['test/batch/iou'] = (inter.sum() / (union.sum() + 1e-10))

                train_step += 1

                iou = inter_meter.sum / (union_meter.sum + 1e-10)
                miou = iou.mean()

                for i, val in enumerate(iou):
                    print('IoU {0}: {1:.2f}'.format(valid_dataset.CLASSES[i], val * 100))
                print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))
                log['test/epoch/loss'] = losses_test / len(dataset)
                log['test/epoch/iou'] = miou.item()
            # time.sleep(60)

            wandb.log(log)
            model.train()

        wandb.finish()
    else:
        print("Evaluating !!! ")
        torch.cuda.set_device(config.gpu)
        model = model.to(device)
        model.eval()

        checkpoint = torch.load(f'/home/hail/Desktop/pan/GCN/PIGNet/model/{config.model_number}/segmentation/{config.dataset}/{config.model_type}/{model_filename}')

        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
        print(model_fname)
        model.load_state_dict(state_dict)
        cmap = loadmat('/home/hail/Desktop/pan/GCN/PIGNet/data/pascal_seg_colormap.mat')['colormap']
        cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

        inter_meter = AverageMeter()
        union_meter = AverageMeter()

        feature_shape = (2048, 33, 33)

        dataset_loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False,
            pin_memory=True, num_workers=config.workers,
            collate_fn=lambda samples: utils_segmentation.make_batch(samples, config.batch_size, feature_shape))

        for i in tqdm(range(len(dataset))):

            inputs, target = dataset[i]
            if inputs==None:
                continue

            inputs = Variable(inputs.to(device))
            outputs , _ = model(inputs.unsqueeze(0))
            _, pred = torch.max(outputs, 1)
            pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
            mask = target.numpy().astype(np.uint8)
            imname = dataset.masks[i].split('/')[-1]
            mask_pred = Image.fromarray(pred)
            mask_pred.putpalette(cmap)
            if config.dataset == 'pascal':
                mask_pred.save(os.path.join('/home/hail/Desktop/pan/GCN/PIGNet/segmentation_result/pascal', imname))
            elif config.dataset == 'cityscapes':
                mask_pred.save(os.path.join('data/cityscapes_val', imname))

            # print('eval: {0}/{1}'.format(i + 1, len(dataset)))

            inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
            inter_meter.update(inter)
            union_meter.update(union)

        iou = inter_meter.sum / (union_meter.sum + 1e-10)
        for i, val in enumerate(iou):
            print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
        print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))
        return iou.mean() * 100

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Load configuration from config.yaml")
    parser.add_argument("--config", type = str, default = "/home/hail/Desktop/pan/GCN/PIGNet/config_segmentation.yaml", help = "path to config.yaml")
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

    if config.mode == "train":

        print("-- Starting Traning Mode --")
        
        main(config)
        
    elif config.mode == "infer":
        print("-- Starting Infer Mode --")
        
        path = f"/home/hail/Desktop/pan/GCN/PIGNet/model/{config.model_number}/segmentation/{config.dataset}/{config.model_type}"
                
        try:
            model_list = sorted(os.listdir(path))
            print(f"[INFO] Found {len(model_list)} models in '{path}'")

        except FileNotFoundError:
            print(f"[ERROR] Model directory not found at '{path}'")
            exit()
    
        zoom_factor = [0.1 , 0.5 , 1 , 1.5 , 2] # zoom in, out value 양수면 줌 음수면 줌아웃

        overlap_percentage = [0,0.1 , 0.2 , 0.3 , 0.5] #겹치는 비율 0~1 사이 값으로 0.8 이상이면 shape 이 안맞음

        pattern_repeat_count = [1,3,6,9,12] # 반복 횟수 2이면 2*2

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
                    iter_config.infer_params.model_filename = name
                    iter_config.infer_params.process_type = process_key
                    iter_config.factor = float(factor_value)

                    print("-" * 60)
                    print(f"Testing model: {name} | Process: {process_key} | Factor: {factor_value}")
                    print("\n")
    
                    accuracy = main(iter_config)

                    if accuracy is not None:
                        output_dict[name][process_key].append(accuracy)
                        
        print("\n--- Inference Results Summary ---")

        # 결과를 보기 좋게 DataFrame으로 변환하여 저장
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
        
        # 열(column)의 이름을 깔끔하게 정리합니다.
        df_wide.rename_axis(columns=None, inplace=True)

        # 3. 변환된 와이드 포맷의 DataFrame을 CSV 파일로 저장합니다.
        output_filename = f"output_wide_{config.model_number}_{config.model_type}_{config.dataset}.csv"
        df_wide.to_csv(output_filename, index=False)
        
        print(f"\n[SUCCESS] Reshaped results saved to '{output_filename}'")
        print(df_wide)

    else:
        print(f"[ERROR] Unknown mode: '{config.mode}'. Please set mode to 'train' or 'infer' in '{cli_args.config}'")


# 텐서 크기에서 중심 좌표를 자동으로 설정 (H/2, W/2)
# H, W = layer_outputs.shape[2], layer_outputs.shape[3]
# center_x, center_y = H // 2, W // 2
#
# 중심점 feature 벡터 (512차원)
# center_vector = layer_outputs[0, :, center_x, center_y]
#
# 각 거리에 대해 코사인 유사도 계산
# for distance in distances:
#     coords = get_coords_by_distance(center_x, center_y, distance, H, W)  # 거리별 좌표 구하기
#     cos_sims = calculate_cosine_similarity(coords, center_vector, layer_outputs)  # 유사도 계산
#     mean_cos_sim = sum(cos_sims) / len(cos_sims)  # 평균 코사인 유사도 계산
#     index = distances.index(distance)
#     distances_sum[index] += mean_cos_sim

# for idx, model in enumerate(pixel_similarity_value):
#    for idx_, data_ in enumerate(model):
#        pixel_similarity_value[idx][idx_]=data_/count