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
from make_classification_dataset import get_dataset
from make_classification_model import get_model

warnings.filterwarnings("ignore")

def main(config):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    print(f"cuda available : {device}")
    
    is_training = (config.mode == "train")

    if not is_training:
   
        # checkpoint full name
        model_filename = config.infer_params.model_filename

        # model name
        m_name = re.search(fr"classification_(.*?)_{config.backbone}", model_filename)
   
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

    model_fname = f'model/{config.model_number}/classification/{config.dataset}/{config.model_type}/classification_{config.model}_{config.backbone}_{config.model_type}_{config.dataset}_v3.pth'

    # define model, dataset
    dataset, dataset_loader, valid_dataset = get_dataset(config)
    model = get_model(config, dataset)

    if config.mode == "train":
        
        # wandb init
        wandb_run_name = f'{config.model}_{config.backbone}_{config.model_type}_embed{config.embedding_size}_nlayer{config.n_layer}_{config.exp}_{config.dataset}'

        config_dict = {k: (vars(v) if isinstance(v, argparse.Namespace) else v) for k, v in vars(config).items()}
        wandb.init(project='gcn_classification', name=wandb_run_name, config=config_dict)


        print("-" * 60)
        print("\n")

        print("Training !!! ")

        print("\n")
        print("-" * 60)

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
        
        if config.model!='swin' and config.model!='vit' :
            backbone_params = (
                        list(model.module.conv1.parameters()) +
                        list(model.module.bn1.parameters()) + 
                        list(model.module.layer1.parameters()) +
                        list(model.module.layer2.parameters()) +
                        list(model.module.layer3.parameters()) +
                        list(model.module.layer4.parameters()))

            if config.model == "PIGNet_GSPonly_classification" or config.model=="PIGNet_classification":
                last_params = list(model.module.pyramid_gnn.parameters())
                optimizer = optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, backbone_params)},
                    {'params': filter(lambda p: p.requires_grad, last_params)}

                ],lr=config.base_lr, momentum=0.9, weight_decay=0.0001)
            
            elif config.model == "ASPP":
                last_params = list(model.module.aspp.parameters())
                optimizer = optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, backbone_params)},
                    {'params': filter(lambda p: p.requires_grad, last_params)}

                ],lr=config.base_lr, momentum=0.9, weight_decay=0.0001)

            else:
                last_params = list(model.module.linear.parameters())
                optimizer = optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, backbone_params)},
                    {'params': filter(lambda p: p.requires_grad, last_params)}

                ],
                    lr=config.base_lr, momentum=0.9, weight_decay=0.0001)

        elif config.model == 'vit':

            # 백본이 아닌 나머지 파라미터만 학습 가능하게 설정
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9,
                                  weight_decay=0.0001)

        else:# swin
            last_params = list(model.module.parameters())
            optimizer = optim.SGD([
                {'params': filter(lambda p: p.requires_grad, last_params)}

            ], lr=0.001, momentum=0.9, weight_decay=0.0001)

        feature_shape = (2048,33,33)

        collate_fn = partial(utils_classification.make_batch_fn, batch_size=config.batch_size, feature_shape=feature_shape)

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

        for epoch in (range(start_epoch, config.epochs)):
            print("EPOCHS : ", epoch+1," / ",config.epochs)

            loss_sum = 0
            acc_sum = 0
            cnt = 0

            for inputs, target in tqdm(iter(dataset_loader)):
                log = {}
                cur_iter = epoch * len(dataset_loader) + cnt
                lr = config.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
                optimizer.param_groups[0]['lr'] = lr
                # if args.model == "deeplab":
                inputs = Variable(inputs.to(device))
                target = Variable(target.to(device)).long()

                if config.model == 'swin' or config.model == 'vit':
                    outputs = model(inputs)

                else:
                    outputs , _ = model(inputs)

                outputs_flat = outputs.view(-1, outputs.size(-1))  # 배치 크기 유지하고 마지막 차원을 평탄화

                # 크로스 엔트로피 손실 계산
                loss = F.cross_entropy(outputs_flat, target)
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    pdb.set_trace()
                losses.update(loss.item(), config.batch_size)
                loss_sum += loss.item()
                loss.backward()
                #print('batch_loss: {:.5f}'.format(loss))

                # calculate accuracy
                _, predicted = outputs.max(1)
                total = target.size(0)
                correct = predicted.eq(target).sum().item()
                accuracy = 100 * correct / total
                acc_sum += accuracy
                log['train/batch/accuracy'] = accuracy
                log['train/batch/loss'] = loss.to('cpu')
                #log['train/temp'] = get_cpu_temperature()
                wandb.log(log)


                cnt+=1
                train_step += 1

                optimizer.step()
                optimizer.zero_grad()

                # if train_step % 30 == 0:
                #     time.sleep(15)

            log = {}
            loss_avg = loss_sum/len(dataset_loader)
            acc_avg = acc_sum/len(dataset_loader)
            log['train/epoch/loss'] = loss_avg
            log['train/epoch/accuracy'] = acc_avg

            wandb.log(log)

            loss_list.append(loss_avg)
            print('epoch: {0}\t'
                  'lr: {1:.6f}\t'
                  'loss: {2:.4f}'.format(
                epoch + 1, lr,loss_avg))



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
                patience +=1
                print("patience : ",patience)

                if patience > 50:
                    break
            # time.sleep(150)
            print("Evaluating !!! ")

            with torch.no_grad():
                model.eval()
                losses_test = 0.0
                correct = 0
                total = 0

                for i, (inputs, labels) in enumerate(tqdm(valid_dataset)):
                    inputs = inputs.to(device)
                    labels = torch.tensor(labels).to(device)

                    if config.model == 'swin' or config.model =='vit':
                        outputs = model(inputs)

                    else:
                        outputs, _ = model(inputs)

                    loss = criterion(outputs, labels)

                    losses_test += loss.item()

                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                accuracy = 100 * correct / total
                print('Accuracy: {:.2f}%'.format(accuracy))

                # train_log = train_log.append({"epoch": epoch, "train_loss":  losses_test / len(valid_dataset), "train_accuracy": accuracy},
                #                              ignore_index=True)


                log['test/epoch/loss'] = losses_test / len(valid_dataset)
                log['test/epoch/accuracy'] = accuracy

            wandb.log(log)
            model.train()

        # train_log.to_csv("training_log.csv", index=False)

    else:
        print("-" * 60)
        print("Evaluating !!! ")
        print("-" * 60)

        torch.cuda.set_device(config.gpu)
        model = model.to(device)
        model.eval()

        checkpoint = torch.load(f'/home/hail/Desktop/HDD/pan/GCN/PIGNet/model/{config.model_number}/classification/{config.dataset}/{config.model_type}/{model_filename}')
        
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
        model.load_state_dict(state_dict)

        with torch.no_grad():
            model.eval()
            losses_test = 0.0
            correct = 0
            total = 0

            distances = [1, 2, 3, 4]
            distances_sum= [ 0 for _ in range(len(distances))]

            for i, (inputs, labels) in enumerate(tqdm(valid_dataset)):

                # import matplotlib.pyplot as plt
                # input_image = inputs.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                # plt.figure(figsize=(6, 6))
                # plt.imshow(input_image)  # 원본 데이터 그대로 출력 (float32 지원)
                # plt.title('Input Image (Original)')
                # plt.axis('off')
                # plt.show()

                inputs = inputs.to(device)
                labels = torch.tensor(labels).to(device)

                if config.model == "vit":

                    outputs = model(inputs)

                else:
                    outputs, layer_outputs = model(inputs)

                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            if config.model != "vit":
                # 텐서 크기에서 중심 좌표를 자동으로 설정 (H/2, W/2)
                H, W = layer_outputs.shape[2], layer_outputs.shape[3]
                center_x, center_y = H // 2, W // 2
                distances = [1, 2, 3, 4]
                # 중심점 feature 벡터 (512차원)
                center_vector = layer_outputs[0, :, center_x, center_y]
                similarities = {distance: [] for distance in distances}

                for distance in distances:
                    x_indices = [max(center_x - distance, 0), min(center_x + distance, H - 1)]
                    y_indices = [max(center_y - distance, 0), min(center_y + distance, W - 1)]

                    for x in range(x_indices[0], x_indices[1] + 1):
                        for y in range(y_indices[0], y_indices[1] + 1):
                            if (x, y) != (center_x, center_y):
                                neighbor_vector = layer_outputs[0, :, x, y]
                                cosine_similarity = F.cosine_similarity(center_vector.unsqueeze(0),
                                                                        neighbor_vector.unsqueeze(0))
                                similarities[distance].append(cosine_similarity.item())

                # 결과 출력
                for distance, values in similarities.items():
                    for idx, sim in enumerate(values):
                        print(f"Distance: {distance}, Cosine Similarity: {sim:.4f}")
                # 각 거리별 평균 유사도 계산
                average_similarities = {distance: sum(values) / len(values) if values else 0 for distance, values in
                                        similarities.items()}

                # # 선 그래프 그리기
                # plt.figure(figsize=(10, 6))
                # plt.plot(average_similarities.keys(), average_similarities.values(), marker='o')
                # plt.title('Average Cosine Similarity by Distance')
                # plt.xlabel('Distance')
                # plt.ylabel('Average Cosine Similarity')
                # plt.xticks(distances)
                # plt.grid(True)
                # save_path = os.path.join("./eval_graph" , f"{args.model}_{args.backbone}_{args.scratch}.png")
                # plt.savefig(save_path , dpi = 300 , bbox_inches = "tight")
                # plt.show()

            accuracy = 100 * correct / total
            print('Accuracy: {:.2f}%'.format(accuracy))
            return accuracy

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Load configuration from config.yaml")
    parser.add_argument("--config", type = str, default = "/home/hail/Desktop/HDD/pan/GCN/PIGNet/config_classification.yaml", help = "path to config.yaml")
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
        
        path = f"/home/hail/Desktop/HDD/pan/GCN/PIGNet/model/{config.model_number}/classification/{config.dataset}/{config.model_type}"
                
        try:
            model_list = sorted(os.listdir(path))
            print(f"[INFO] Found {len(model_list)} models in '{path}'")

        except FileNotFoundError:
            print(f"[ERROR] Model directory not found at '{path}'")
            exit()

        zoom_ratio = [0.1, 0.5, 1, 1.5, 2]

        rotate_degree = [180, 150, 120, 90, 60, 45, 30, 15, 0, -15, -30, -45, -60, -90, -120, -150, -180]

        # zoom_ratio = [1]

        # rotate_degree = [0]

        # process_dict = {"zoom": zoom_ratio, "rotate": rotate_degree}

        process_dict = {"zoom": zoom_ratio, "rotate": rotate_degree}

        output_dict = {model_name: {"zoom": [], "rotate": []} for model_name in model_list}

        for name in model_list:
            for process_key, factor_list in process_dict.items():
                for factor_value in factor_list:

                    # 루프마다 config 객체를 깊은 복사하여 수정 (원본 config는 유지)

                    iter_config = copy.deepcopy(config)
                    iter_config.infer_params.model_filename = name
                    iter_config.infer_params.process_type = process_key
                    iter_config.factor = float(factor_value)

                    print("-" * 60)
                    print(f"Testing model: {name} | Process: {process_key} | Factor: {factor_value}")
                    print("\n")

                    # 수정된 config 객체로 main 함수 호출
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

