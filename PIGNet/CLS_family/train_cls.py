import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import wandb
import warnings
import yaml
import os
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm.auto import tqdm
from utils import AverageMeter
from GCN.PIGNet.CLS_family.cls_dataset import get_dataset
from GCN.PIGNet.CLS_family.cls_models import get_model

warnings.filterwarnings("ignore")

def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    print(f"cuda available : {device}")

    config.model = config.model
    print(f"Mode: {config.mode} | Model: {config.model} | Dataset: {config.dataset} | Device: {device}")
    
    loss_list = []
    if config.backbone == "resnet50":
        num=50
    elif config.backbone == "resnet101":
        num=101

    model_fname = f'model_{num}/{config.model_number}/classification/{config.dataset}/{config.model_type}/classification_{config.model}_{config.backbone}_{config.model_type}_{config.dataset}_v3.pth'

    # define model, dataset
    dataset, dataset_loader, valid_dataset = get_dataset(config)
    model = get_model(config, dataset)

    if config.mode == "train":
            
        # wandb init
        # wandb_run_name = f'{config.model}_{config.backbone}_{config.model_type}_embed{config.embedding_size}_nlayer{config.n_layer}_{config.exp}_{config.dataset}'

        config_dict = {k: (vars(v) if isinstance(v, argparse.Namespace) else v) for k, v in vars(config).items()}
        # wandb.init(project='gcn_classification', name=wandb_run_name, config=config_dict)

        print("-" * 60)
        print("\n")
        print("Training !!! ")
        print("\n")
        print("-" * 60)

        criterion = nn.CrossEntropyLoss(ignore_index=255)
        model = model.to(device)
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
                        list(model.conv1.parameters()) +
                        list(model.bn1.parameters()) + 
                        list(model.layer1.parameters()) +
                        list(model.layer2.parameters()) +
                        list(model.layer3.parameters()) +
                        list(model.layer4.parameters()))

            if config.model == "PIGNet_GSPonly_classification" or config.model=="PIGNet_classification":
                last_params = list(model.pyramid_gnn.parameters())
                optimizer = optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, backbone_params)},
                    {'params': filter(lambda p: p.requires_grad, last_params)}

                ],lr=config.base_lr, momentum=0.9, weight_decay=0.0001)
            
            elif config.model == "ASPP":
                last_params = list(model.aspp.parameters())
                optimizer = optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, backbone_params)},
                    {'params': filter(lambda p: p.requires_grad, last_params)}

                ],lr=config.base_lr, momentum=0.9, weight_decay=0.0001)

            else:
                last_params = list(model.linear.parameters())
                optimizer = optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, backbone_params)},
                    {'params': filter(lambda p: p.requires_grad, last_params)}

                ],
                    lr=config.base_lr, momentum=0.9, weight_decay=0.0001)

        elif config.model == 'vit':

            # 백본이 아닌 나머지 파라미터만 학습 가능하게 설정
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9,
                                weight_decay=0.0001)

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
                inputs = Variable(inputs.to(device))
                target = Variable(target.to(device)).long()

                if config.model == 'swin' or config.model == 'vit':
                    outputs = model(inputs)
                else:
                    outputs , _, _ = model(inputs)
                outputs_flat = outputs.view(-1, outputs.size(-1))  # 배치 크기 유지하고 마지막 차원을 평탄화

                loss = F.cross_entropy(outputs_flat, target)
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    pdb.set_trace()
                losses.update(loss.item(), config.batch_size)
                loss_sum += loss.item()
                loss.backward()

                # calculate accuracy
                _, predicted = outputs.max(1)
                total = target.size(0)
                correct = predicted.eq(target).sum().item()
                accuracy = 100 * correct / total
                acc_sum += accuracy
                log['train/batch/accuracy'] = accuracy
                log['train/batch/loss'] = loss.to('cpu')
                # wandb.log(log)

                cnt+=1
                train_step += 1

                optimizer.step()
                optimizer.zero_grad()

            log = {}
            loss_avg = loss_sum/len(dataset_loader)
            acc_avg = acc_sum/len(dataset_loader)
            log['train/epoch/loss'] = loss_avg
            log['train/epoch/accuracy'] = acc_avg

            # wandb.log(log)

            loss_list.append(loss_avg)
            print('epoch: {0}\t'
                'lr: {1:.6f}\t'
                'loss: {2:.4f}'.format(
                epoch + 1, lr,loss_avg))

            if best_loss > loss_avg:
                best_loss = loss_avg
                patience = 0

                # 모델 저장 폴더가 없으면 생성
                model_dir = os.path.dirname(model_fname)
                os.makedirs(model_dir, exist_ok=True)

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

            print("Evaluating !!! ")
            
            with torch.no_grad():
                model.eval()
                losses_test = 0.0
                correct = 0
                total = 0

                # assert isinstance(valid_dataset.sampler, torch.utils.data.SequentialSampler)
                
                for i, (inputs, labels) in enumerate(tqdm(valid_dataset)):
                    inputs = inputs.to(device)
                    labels = torch.tensor(labels).to(device)

                    if config.model == 'swin' or config.model =='vit':
                        outputs = model(inputs)

                    else:
                        outputs, _, _ = model(inputs)

                    loss = criterion(outputs, labels)

                    losses_test += loss.item()

                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                accuracy = 100 * correct / total
                print('Accuracy: {:.2f}%'.format(accuracy))
                log['test/epoch/loss'] = losses_test / len(valid_dataset)
                log['test/epoch/accuracy'] = accuracy

            # wandb.log(log)
            model.train()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Load configuration from config.yaml")
    parser.add_argument("--config", type = str, default = "/home/hail/pan/GCN/PIGNet/config_classification.yaml", help = "path to config.yaml")
    parser.add_argument("--mode", type = str, default = "train", help = "training mode")
    parser.add_argument("--model", type = str, default = "vit", help = "PIGNet_GSPonly_classification, vit, Resnet")
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
    config.mode = cli_args.mode
    config.model = cli_args.model

    print("-- Starting Traning Mode --")
    main(config)
        