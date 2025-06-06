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

warnings.filterwarnings("ignore")

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



parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true',
                    help='training mode')
parser.add_argument('--exp', type=str,default="bn_lr7e-3",
                    help='name of experiment')
parser.add_argument('--gpu', type=int, default=0,
                    help='test time gpu device id')
parser.add_argument('--backbone', type=str, default='resnet50',
                    help='resnet50')
parser.add_argument('--dataset', type=str, default='pascal',
                    help='pascal or cityscapes')
parser.add_argument('--groups', type=int, default=None,
                    help='num of groups for group normalization')
parser.add_argument('--epochs', type=int, default=30,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.007,
                    help='base learning rate')
parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='train from scratch')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='freeze batch normalization parameters')
parser.add_argument('--weight_std', action='store_true', default=False,
                    help='weight standardization')
parser.add_argument('--beta', action='store_true', default=False,
                    help='resnet101 beta')
parser.add_argument('--crop_size', type=int, default=513,
                    help='image crop size')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4,
                    help='number of model loading workers')
parser.add_argument('--model', type=str, default="deeplab",
                    help='model name')
parser.add_argument('--process_type', type=str, default="zoom",
                    help='process_type')

parser.add_argument('--infer' , action = "store_true")

parser.add_argument("--trained_model" , default="scratch" , required = False , help = "scratch or pretrained")

args = parser.parse_args()


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

def main(process_type , factor , model_name):

    print(f"type : {type(model_name)} , model_name : {model_name}")

    m_name = re.search(r"(.*?)_resnet50" , model_name)

    if m_name:
        m_name = m_name.group(1)

    # make fake ar
    args = argparse.Namespace()
    args.dataset = "pascal" # cityscape pascal
    args.model = m_name #PIGNet PIGNet_GSPonly  Mask2Former ASPP
    args.backbone = "resnet50" # resnet[50 , 101]
    args.scratch = True
    args.train = False
    args.workers = 4
    args.epochs = 50
    args.batch_size = 16
    args.crop_size = 513
    args.base_lr = 0.007
    args.last_mult = 1.0
    args.groups = None
    args.freeze_bn = False
    args.weight_std = False
    args.beta = False
    args.resume = None
    args.exp = "bn_lr7e-3"
    args.gpu = 0
    args.embedding_size = 21
    args.n_layer = 12
    args.n_skip_l = 3
    args.factor = factor
    args.process_type = process_type

    if args.train:
        if args.scratch == False:
            wandb.init(project='gcn_segmentation', name=args.model+'_'+args.backbone+ '_pretrain' + '_embed' + str(args.embedding_size) +'_nlayer' + str(args.n_layer) + '_'+args.exp+'_'+str(args.dataset),
                        config=args.__dict__)
        else:
            wandb.init(project='gcn_segmentation', name=args.model+'_'+args.backbone+ '_scratch' + '_embed' + str(args.embedding_size) +'_nlayer' + str(args.n_layer) + '_'+args.exp+'_'+str(args.dataset),
                        config=args.__dict__)

    # if is cuda available device
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    print("ags",args.model,args.dataset)
    print("cuda available",args.device)

    loss_list = []
    # assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    if args.scratch == True:
        model_fname = 'model/segmentation/{0}_{1}_{2}_scratch_v3.pth'.format(
            args.model,args.backbone, args.dataset,args.embedding_size)
    else: # pretrain
        model_fname = 'model/segmentation/{0}_{1}_{2}_pretrain_v3.pth'.format(
            args.model,args.backbone, args.dataset,args.embedding_size)

    if args.dataset == 'pascal':
        if args.train:

            print("train dataset")
            dataset = VOCSegmentation('/home/hail/Desktop/pan/GCN/gcn-ha/PIGNet_ha/data/VOCdevkit',
                                      train=args.train, crop_size=args.crop_size)
            valid_dataset = VOCSegmentation('/home/hail/Desktop/pan/GCN/gcn-ha/PIGNet_ha/data/VOCdevkit',
                                            train=not (args.train), crop_size=args.crop_size)
        else:

            if args.process_type != None:
                print(args.process_type)
                dataset = VOCSegmentation('/home/hail/Desktop/pan/GCN/gcn-ha/PIGNet_ha/data/VOCdevkit',
                                                train=args.train, crop_size=args.crop_size,
                                                process=args.process_type, process_value=args.factor,
                                                overlap_percentage=args.factor,
                                                pattern_repeat_count=args.factor)
            else:
                dataset = VOCSegmentation('/home/hail/Desktop/pan/GCN/gcn-ha/PIGNet_ha/data/VOCdevkit',
                                                train=args.train, crop_size=args.crop_size,
                                                process=None, process_value=args.factor,
                                                overlap_percentage=args.factor,
                                                pattern_repeat_count=args.factor)

    elif args.dataset == 'cityscape':
        if args.train:
            print("train dataset cityscape")

            dataset = Cityscapes('/home/hail/Desktop/pan/GCN/gcn-ha/PIGNet_ha/data/cityscape',
                                 train=args.train, crop_size=args.crop_size)

            valid_dataset = Cityscapes('/home/hail/Desktop/pan/GCN/gcn-ha/PIGNet_ha/data/cityscape',
                                 train=not (args.train), crop_size=args.crop_size)

        else: # val
            if args.process_type != None:
                print(args.process_type)
                dataset = Cityscapes('/home/hail/Desktop/pan/GCN/gcn-ha/PIGNet_ha/data/cityscape',
                                          train=args.train, crop_size=args.crop_size,
                                          process=args.process_type, process_value=args.factor,
                                          overlap_percentage=args.factor,
                                          pattern_repeat_count=args.factor)
            else:
                dataset = Cityscapes('/home/hail/Desktop/pan/GCN/gcn-ha/PIGNet_ha/data/cityscape',
                                          train=args.train, crop_size=args.crop_size,
                                          process=None, process_value=args.factor,
                                          overlap_percentage=args.factor,
                                          pattern_repeat_count=args.factor)

    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.backbone in ["resnet50","resnet101","resnet152"]:

        if args.model == "PIGNet_GSPonly":
            model = getattr(PIGNet_GSPonly, args.backbone)(
                pretrained=(not args.scratch),
                num_classes=len(dataset.CLASSES),
                num_groups=args.groups,
                weight_std=args.weight_std,
                beta=args.beta,
                embedding_size = args.embedding_size,
                n_layer = args.n_layer,
                n_skip_l = args.n_skip_l)

        elif args.model == "PIGNet":
            model = getattr(PIGNet, args.backbone)(
                pretrained=(not args.scratch),
                num_classes=len(dataset.CLASSES),
                num_groups=args.groups,
                weight_std=args.weight_std,
                beta=args.beta,
                embedding_size = args.embedding_size,
                n_layer = args.n_layer,
                n_skip_l = args.n_skip_l)

        elif args.model == "ASPP":
            model = getattr(ASPP, args.backbone)(
                pretrained=(not args.scratch),
                num_classes=len(dataset.CLASSES),
                num_groups=args.groups,
                weight_std=args.weight_std,
                beta=args.beta,
                embedding_size = args.embedding_size,
                n_layer = args.n_layer,
                n_skip_l = args.n_skip_l)

        elif args.model == "Mask2Former":
            model = Mask2Former()

    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    size_in_bytes = model_size(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print number of parameters
    print(f"Number of parameters: {num_params / (1000.0 ** 2): .3f} M")

    # num_params_gnn = sum(p.numel() for p in model.pyramid_gnn.parameters() if p.requires_grad)
    # print(f"Number of GNN parameters: {num_params_gnn / (1000.0 ** 2): .3f} M")

    print(f"Entire model size: {size_in_bytes / (1024.0 ** 3): .3f} GB")

    if args.train:
        print("Training !!! ")

        criterion = nn.CrossEntropyLoss(ignore_index=255)
        model = nn.DataParallel(model).to(args.device)
        model.train()

        if args.freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    print("12345")
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

        if args.model != "Mask2Former":
            backbone_params = (
                    list(model.module.conv1.parameters()) +
                    list(model.module.bn1.parameters()) +
                    list(model.module.layer1.parameters()) +
                    list(model.module.layer2.parameters()) +
                    list(model.module.layer3.parameters()) +
                    list(model.module.layer4.parameters()))

        if args.model == "PIGNet_GSPonly" or args.model=="PIGNet":
            last_params = list(model.module.pyramid_gnn.parameters())

        elif args.model == "ASPP":
            last_params = list(model.module.aspp.parameters())

        else: # Masktoformer
            backbone_params = list(model.module.backbone.parameters())
            last_params = list(model.module.transformer_decoder.parameters())

        optimizer = optim.SGD([
            {'params': filter(lambda p: p.requires_grad, backbone_params)},
            {'params': filter(lambda p: p.requires_grad, last_params)}

        ],
            lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
        feature_shape = (2048, 33, 33)

        collate_fn = partial(make_batch_fn, batch_size=args.batch_size, feature_shape=feature_shape)

        dataset_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.train,
            pin_memory=True,
            num_workers=args.workers,
            collate_fn=collate_fn
        )

        max_iter = args.epochs * len(dataset_loader)
        losses = AverageMeter()
        start_epoch = 0

        if args.resume:
            if os.path.isfile(args.resume):
                print('=> loading checkpoint {0}'.format(args.resume))
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('=> loaded checkpoint {0} (epoch {1})'.format(
                    args.resume, checkpoint['epoch']))
            else:
                print('=> no checkpoint found at {0}'.format(args.resume))

        best_loss = 1e+10
        patience = 0

        train_step = 0

        for epoch in range(start_epoch, args.epochs):
            print("EPOCHS : ", epoch + 1, " / ", args.epochs)

            loss_sum = 0
            cnt = 0

            for inputs, target in tqdm(iter(dataset_loader)):
                log = {}
                cur_iter = epoch * len(dataset_loader) + cnt
                lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
                optimizer.param_groups[0]['lr'] = lr
                # if args.model == "deeplab":
                optimizer.param_groups[1]['lr'] = lr * args.last_mult
                inputs = Variable(inputs.to(args.device))
                target = Variable(target.to(args.device)).long()
                outputs , _ = model(inputs)
                outputs = outputs.float()
                loss = criterion(outputs, target)
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    pdb.set_trace()
                losses.update(loss.item(), args.batch_size)
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
                    inputs = Variable(inputs.to(args.device))
                    target = Variable(target.to(args.device)).long()
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
        torch.cuda.set_device(args.gpu)
        model = model.to(args.device)
        model.eval()

        if args.scratch != True: # pretrained
            checkpoint = torch.load(f"/home/hail/Desktop/pan/GCN/gcn-ha/PIGNet_ha/model/segmentation/{args.dataset}/pretrained/{model_name}")
        else:
            checkpoint = torch.load(f"/home/hail/Desktop/pan/GCN/gcn-ha/PIGNet_ha/model/segmentation/{args.dataset}/scratch/{model_name}")

        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
        print(model_fname)
        model.load_state_dict(state_dict)
        cmap = loadmat('/home/hail/Desktop/pan/GCN/gcn-ha/PIGNet_ha/data/pascal_seg_colormap.mat')['colormap']
        cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

        inter_meter = AverageMeter()
        union_meter = AverageMeter()

        feature_shape = (2048, 33, 33)

        dataset_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            pin_memory=True, num_workers=args.workers,
            collate_fn=lambda samples: make_batch(samples, args.batch_size, feature_shape))

        for i in tqdm(range(len(dataset))):

            inputs, target = dataset[i]
            if inputs==None:
                continue

            inputs = Variable(inputs.to(args.device))
            outputs , _ = model(inputs.unsqueeze(0))
            _, pred = torch.max(outputs, 1)
            pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
            mask = target.numpy().astype(np.uint8)
            imname = dataset.masks[i].split('/')[-1]
            mask_pred = Image.fromarray(pred)
            mask_pred.putpalette(cmap)
            if args.dataset == 'pascal':
                mask_pred.save(os.path.join('/home/hail/Desktop/pan/GCN/gcn-ha/PIGNet_ha/segmentation_result/pascal', imname))
            elif args.dataset == 'cityscapes':
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

    path = f"/home/hail/Desktop/pan/GCN/gcn-ha/PIGNet_ha/model/segmentation/{args.dataset}/{args.trained_model}"
    model_list = sorted(os.listdir(path))

    zoom_factor = [0.1 , 0.5 , 1 , 1.5 , 2] # zoom in, out value 양수면 줌 음수면 줌아웃
    overlap_percentage = [0.1 , 0.2 , 0.3 , 0.5] #겹치는 비율 0~1 사이 값으로 0.8 이상이면 shape 이 안맞음
    pattern_repeat_count = [3,6,9,12] # 반복 횟수 2이면 2*2

    output_dict = {model_name : {"zoom" : [] , "overlap" : [] , "repeat" : []} for model_name in model_list}

    process_dict = {
        "zoom" : zoom_factor , 
        "overlap" : overlap_percentage ,
        "repeat" : pattern_repeat_count
    }

    if args.infer == True: # inference
        for name in model_list:
            for key , value in process_dict.items():
                for ratio in value:
                    output = main(key , ratio , name)
                    output_dict[name][key].append(output)
        print(output_dict)

        records = []

        for model_name , result_dict in output_dict.items():
            for task , values in result_dict.items():
                for i , val in enumerate(values):
                    records.append({
                        "model" : model_name ,
                        "task" : task ,
                        "index" : i , 
                        "value" : val
                    })

        df = pd.DataFrame(records)
        df.to_csv(f"output_{args.trained_model}_{args.dataset}.csv" , index = False)
    
    else: 
        main(None , None , None)


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