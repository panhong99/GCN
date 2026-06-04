import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import wandb
import warnings
import seg_utils as utils_segmentation
import yaml

from tqdm.auto import tqdm
from utils import AverageMeter, inter_and_union
from functools import partial
from GCN.PIGNet.seg_dataset import get_dataset
from GCN.PIGNet.seg_models import get_model

warnings.filterwarnings("ignore")

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Mode: {config.mode} | Model: {config.model} | Dataset: {config.dataset} | Device: {device}")

    if config.backbone == "resnet50":
        num=50
    elif config.backbone == "resnet101":
        num=101

    feature_shape = (2048, 33, 33)
    collate_fn = partial(utils_segmentation.make_batch_fn, batch_size=config.batch_size, feature_shape=feature_shape)
    dataset, valid_dataset = get_dataset(config)
    dataset_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.workers,
    collate_fn=collate_fn
    )

    model = get_model(config, dataset)
    model.to(device)
    model.train()
    model_fname = f'model_{num}/{config.model_number}/segmentation/{config.dataset}/{config.model_type}/{config.model}_{config.backbone}_{config.model_type}_{config.dataset}_{config.n_layer}.pth'

    if config.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                print("12345")
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                
    if config.model != "Mask2Former":
        backbone_params = (
            list(model.conv1.parameters()) +
            list(model.bn1.parameters()) +
            list(model.layer1.parameters()) +
            list(model.layer2.parameters()) +
            list(model.layer3.parameters()) +
            list(model.layer4.parameters()))

    if config.model == "PIGNet_GSPonly" or config.model=="PIGNet":
        last_params = list(model.pyramid_gnn.parameters())

    elif config.model == "ASPP":
        last_params = list(model.aspp.parameters())

    else: # Mask2Former
        backbone_params = list(model.backbone.parameters())
        last_params = list(model.transformer_decoder.parameters())

    max_iter = config.epochs * len(dataset_loader)
    losses = AverageMeter()
    start_epoch = 0

    loss_list = []
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.SGD([
    {'params': filter(lambda p: p.requires_grad, backbone_params)},
    {'params': filter(lambda p: p.requires_grad, last_params)}
    ],lr=config.base_lr, momentum=0.9, weight_decay=1e-4)

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

    # wandb.init(project = "gcn_segmentation", name=config.model+"_"+config.backbone+"_"+str(config.model_type)+"_embed"+str(config.embedding_size)+"_nlayer"+str(config.n_layer)+"_"+config.exp+"_"+str(config.dataset),
    #     config=config.__dict__)

    for epoch in range(start_epoch, config.epochs):
        print("EPOCHS : ", epoch + 1, " / ", config.epochs)
        loss_sum = 0
        cnt = 0

        data_iterator = tqdm(iter(dataset_loader))
                            
        for inputs, target in data_iterator:
            log = {}
            cur_iter = epoch * len(dataset_loader) + cnt
            lr = config.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * config.last_mult
            inputs = inputs.to(device)
            target = target.to(device).long()
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
            log['train/batch/loss'] = loss.to('cpu')
            cnt += 1

            train_step += 1
            optimizer.step()
            optimizer.zero_grad()

        loss_avg = loss_sum / len(dataset_loader)
        log['train/epoch/loss'] = loss_avg

        # wandb.log(log)

        loss_list.append(loss_avg)
        print('epoch: {0}\t'
            'lr: {1:.6f}\t'
            'loss: {2:.4f}'.format(
            epoch + 1, lr, loss_avg))

        if best_loss > loss_avg:
            best_loss = loss_avg
            patience = 0
            state_dict = model.state_dict()                
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

        print("Evaluating !!! ")

        inter_meter = AverageMeter()
        union_meter = AverageMeter()

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
            
            eval_iterator = tqdm(valid_loader)

            for inputs, target in eval_iterator:
                inputs = inputs.to(device)
                target = target.to(device).long()
                outputs = model(inputs)
                
                if config.model == "Mask2Former":
                    outputs = outputs.float()
                else:
                    outputs = outputs[0].float()

                _, pred = torch.max(outputs, 1)
                pred = pred.data.cpu().numpy().astype(np.uint8)
                mask = target.cpu().numpy().astype(np.uint8)
                for p, m in zip(pred, mask):
                    inter, union = inter_and_union(p, m, len(valid_dataset.CLASSES))
                    inter_meter.update(inter)
                    union_meter.update(union)
                    loss = criterion(outputs, target)
                    losses_test += loss.item() * inputs.size(0)

            train_step += 1

            final_inter = inter_meter.sum
            final_union = union_meter.sum                    
            iou = final_inter / (final_union + 1e-10)
            miou = iou.mean()                    
            total_losses_test = losses_test
            total_samples = len(valid_dataset)
            
            for i, val in enumerate(iou):
                print('IoU {0}: {1:.2f}'.format(valid_dataset.CLASSES[i], val * 100))
            print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))
            
            avg_valid_loss = total_losses_test / total_samples 
            log['test/epoch/loss'] = avg_valid_loss
            log['test/epoch/iou'] = miou.item()
        
        # wandb.log(log)
        model.train()
        
    # wandb.finish()

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Load configuration from config.yaml")
    parser.add_argument("--config", type = str, default = "/home/hail/pan/GCN/PIGNet/config_segmentation.yaml", help = "path to config.yaml")
    parser.add_argument("--mode", type = str, default = "train", help = "training or evaluation mode")
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
    config.mode = args.mode
    main(config)