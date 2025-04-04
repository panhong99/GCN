import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from scipy.spatial.distance import cosine
from torch.autograd import Variable
from tqdm.auto import tqdm
from torch.utils.data import RandomSampler
import pandas as pd
import os
from torchvision import transforms

from PIGNet.model_code import swin, Classification_resnet, PIGNet_GSPonly_classification

print("Current directory:", os.getcwd())

import torch.nn.functional as F


from PIGNet.pascal import VOCSegmentation
from PIGNet.utils import AverageMeter

from functools import partial
import torchvision
# time.sleep(600)600
def make_batch_fn(samples, batch_size, feature_shape):
    return make_batch(samples, batch_size, feature_shape)


import subprocess
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import wandb


def visualize_compared_features(compared_features):
    plt.imshow(compared_features, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Compared Features')
    plt.show()

# compared에 해당하는 특징 벡터를 시각화

def compute_similarity(feature_map,labels):
    # feature_map의 shape을 가져옴
    batch_size, num_channels, height, width = feature_map.size()


    similarities = [0 for _ in range(4)]
    for channel_idx in range(num_channels):
        # 현재 채널(feature)의 특징 벡터
        channel_features = feature_map[:, channel_idx, :]

        # 채널의 중심(center) 지점
        center_idx = (int(height/2),int(width/2))
        center = channel_features[:, center_idx[0], center_idx[1]]  # 모든 데이터 포인트에 대해 중심 값을 가져옴
        distance_count=0
        # center에서 일정 거리만큼 떨어진 지점들의 코사인 유사도 계산
        distance = 2
        for z in range(4):  # distance가 1부터 4까지 변경 가능
            compared_indices = [(center_idx[0] + dx, center_idx[1] + dy) for dx in range(-distance, distance + 1) for dy
                                in range(-distance, distance + 1)]
            # 중복 제거
            compared_indices = list(set(compared_indices))

            # 이전 거리에 해당하는 인덱스 제외
            if distance > 1:
                previous_indices = [(center_idx[0] + dx, center_idx[1] + dy) for dx in range(-(distance - 1), distance)
                                    for dy
                                    in range(-(distance - 1), distance)]
                # 유효한 인덱스만 남기고 다른 것들 제거
                previous_indices = [(x, y) for (x, y) in previous_indices if 0 <= x < height and 0 <= y < width]
                compared_indices=[(x, y) for (x, y) in compared_indices if 0 <= x < height and 0 <= y < width]
                compared_indices = list(set(compared_indices) - set(previous_indices))



            # 현재 distance에 해당하는 지점들의 특징 벡터 가져오기
            compared = channel_features[:, [idx[0] for idx in compared_indices], [idx[1] for idx in compared_indices]]
            grap_compared = compared.cpu().squeeze()


            similarity = cosine_similarity(center, compared,dim=1)
            similarities[distance_count] += abs(similarity)
            distance_count+=1
            distance *= 2

    similarities_ = [similarity / num_channels for similarity in similarities]
    # 각 채널의 픽셀 유사도의 평균을 반환
    return similarities_

def get_cpu_temperature():
    sensors_output = subprocess.check_output("sensors").decode()
    for line in sensors_output.split("\n"):
        if "Tctl" in line:
            temperature_str = line.split()[1]
            temperature = float(temperature_str[:-3])  # remove "°C" and convert to float
            return temperature

    return None  # in case temperature is not found



parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False,
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

args = parser.parse_args()
def model_size(model):
    total_size = 0
    for param in model.parameters():
        # °¢ ÆÄ¶ó¹ÌÅÍÀÇ ¿ø¼Ò °³¼ö °è»ê
        num_elements = torch.prod(torch.tensor(param.size())).item()
        # ¿ø¼Ò Å¸ÀÔ º°·Î ¹ÙÀÌÆ® Å©±â °è»ê (¿¹: float32 -> 4 bytes)
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


def main():
    # make fake args
    args = argparse.Namespace()
    args.dataset = "imagenet" #CIFAR-10 CIFAR-100    imagenet  pascal
    args.model = "Resnet" #Resnet  PIGNet_GSPonly_classification  vit_b_16  swin
    args.backbone = "resnet101"
    args.workers = 4
    args.epochs = 50
    args.batch_size = 8
    args.train = False
    args.crop_size = 513 #513
    args.base_lr = 0.007
    args.last_mult = 1.0
    args.groups = None
    args.scratch = False
    args.freeze_bn = False
    args.weight_std = False
    args.beta = False
    args.resume = None
    args.exp = "bn_lr7e-3"
    args.gpu = 0
    args.embedding_size = 512
    args.n_layer = 6
    args.n_skip_l = 2
    args.process_type = "repeat"  #zoom overlap repeat
    # if is cuda available device
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    wandb.init(project='pignet_classification', name=args.model+'_'+args.backbone+ '_embed' + str(args.embedding_size) +'_nlayer' + str(args.n_layer) + '_'+args.exp,
                config=args.__dict__)

    loss_data = pd.DataFrame(columns=["train_loss"])
    loss_list = []

    # assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    model_fname = 'model/classification_{0}_{1}_{2}_v3.pth'.format(
        args.model,args.backbone, args.dataset)

    if args.dataset == 'pascal':
        image_size = 513
        dataset = VOCSegmentation('C:/Users/hail/Desktop/ha/data/ADE/VOCdevkit',
                                 train=args.train, crop_size=args.crop_size, process= None)
        valid_dataset = VOCSegmentation('C:/Users/hail/Desktop/ha/data/ADE/VOCdevkit',
                                        train=not (args.train), crop_size=args.crop_size, process= None)

        zoom_factor = 0.5 # zoom in, out value 양수면 줌 음수면 줌아웃
        overlap_percentage = 0.3
        pattern_repeat_count = 3
        process_valid_dataset = VOCSegmentation('C:/Users/hail/Desktop/ha/data/ADE/VOCdevkit',
                                        train=args.train,crop_size=args.crop_size, process= args.process_type,process_value = zoom_factor,overlap_percentage= overlap_percentage,pattern_repeat_count=pattern_repeat_count)
        for i in range(len(process_valid_dataset)):
            img, target = process_valid_dataset[i]
            if img ==None or target ==None:
                continue
    elif args.dataset == 'imagenet':
        # 데이터셋 경로 및 변환 정의
        image_size=224
        data_dir = 'C:/Users/hail/Desktop/ha/data/Imagenet'
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.ImageFolder(root=data_dir+'/train', transform=transform)

        valid_dataset = torchvision.datasets.ImageFolder(root=data_dir+'/val', transform=transform)
        idx2label = []
        cls2label = {}

        import json
        json_file=data_dir+'/imagenet_class_index.json'
        with open(json_file, "r") as read_file:
            class_idx = json.load(read_file)
            idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
        dataset.CLASSES = idx2label

    elif args.dataset == 'CIFAR-100':
        image_size = 32
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 정규화합니다.
        ])

        # CIFAR-100 데이터셋 로드
        dataset = torchvision.datasets.CIFAR100(root='C:/Users/hail/Desktop/ha/model', train=True, download=True, transform=transform)
        valid_dataset = torchvision.datasets.CIFAR100(root='C:/Users/hail/Desktop/ha/model', train=False, download=True, transform=transform)

        dataset.CLASSES=sorted(['beaver', 'dolphin', 'otter', 'seal', 'whale',  # aquatic mammals
                           'aquarium' 'fish', 'flatfish', 'ray', 'shark', 'trout',  # fish
                           'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', # flowers
                           'bottles', 'bowls', 'cans', 'cups', 'plates', # food containers
                           'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', # fruit and vegetables
                           'clock', 'computer' 'keyboard', 'lamp', 'telephone', 'television', # household electrical devices
                           'bed', 'chair', 'couch', 'table', 'wardrobe', # household furniture
                           'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', # insects
                           'bear', 'leopard', 'lion', 'tiger', 'wolf', # large carnivores
                           'bridge', 'castle', 'house', 'road', 'skyscraper', # large man-made outdoor things
                           'cloud', 'forest', 'mountain', 'plain', 'sea', # large natural outdoor scenes
                           'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', # large omnivores and herbivores
                           'fox', 'porcupine', 'possum', 'raccoon', 'skunk', # medium-sized mammals
                           'crab', 'lobster', 'snail', 'spider', 'worm', # non-insect invertebrates
                           'baby', 'boy', 'girl', 'man', 'woman', # people
                           'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', # reptiles
                           'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', # small mammals
                           'maple', 'oak', 'palm', 'pine', 'willow', # trees
                           'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', # vehicles 1
                           'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor' # vehicles 2
                          ])
    elif args.dataset == 'CIFAR-10':
        # Define the desired size
        desired_size = 128
        image_size = 32

        # Calculate the padding needed for zoom out
        padding_zoom_out = (desired_size - 32) // 2

        # Define the transformation pipeline for zoom in and zoom out
        transform_zoom_in = transforms.Compose([
            transforms.Resize((int(desired_size * 2), int(desired_size * 2))),  # Slight enlargement
            transforms.CenterCrop(desired_size) ,
            transforms.ToTensor()# Crop to desired size
        ])

        transform_zoom_in_target = transforms.Compose([
            transforms.Resize(desired_size),  # Resize to desired size
            transforms.CenterCrop(desired_size),  # Crop to desired size
            transforms.ToTensor()  # Convert to tensor
        ])

        transform_zoom_out = transforms.Compose([
            transforms.Pad(padding_zoom_out, fill=0, padding_mode='constant'),  # Pad the image with zeros
            transforms.Resize((desired_size, desired_size)),
            transforms.ToTensor()# Resize to desired size
        ])

        # Rest of your code remains the same, you can apply the desired transformation based on your requirement
        #transform = transform_zoom_in
        #transform_target = transform_zoom_in_target

        transform = transforms.Compose([
            transforms.ToTensor()])

        # CIFAR-100 데이터셋 로드
        dataset = torchvision.datasets.CIFAR10(root='C:/Users/hail/Desktop/ha/model', train=True, download=True,transform=transform)#, transform=transform,target_transform=transform_target)
        valid_dataset = torchvision.datasets.CIFAR10(root='C:/Users/hail/Desktop/ha/model', train=False, download=True,transform=transform)#, transform=transform,target_transform=transform_target)
        dataset.CLASSES =['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.backbone in ["resnet50","resnet101","resnet152"]:
        if args.model == "PIGNet_GSPonly_classification":
            model = getattr(PIGNet_GSPonly_classification, args.backbone)(
                pretrained=(not args.scratch),
                num_classes=len(dataset.CLASSES),
                num_groups=args.groups,
                weight_std=args.weight_std,
                beta=args.beta,
                embedding_size = args.embedding_size,
                n_layer = args.n_layer,
                n_skip_l = args.n_skip_l)
            print("model PIGNet_GSPonly_classification")
        elif args.model == "Resnet":
            print("Classification_resnet model load")
            model = getattr(Classification_resnet, args.backbone)(
                pretrained=(not args.scratch),
                num_classes=len(dataset.CLASSES),
                num_groups=args.groups,
                weight_std=args.weight_std,
                beta=args.beta,
                embedding_size=args.embedding_size,
                n_layer=args.n_layer,
                n_skip_l=args.n_skip_l
                )
        elif args.model == 'vit_b_16':
            print("main")
        elif args.model == 'swin':
            model = swin.SwinTransformer(img_size=image_size, num_classes=len(dataset.CLASSES), window_size=4) #window_size =4 cifar   img_size=

    else:

        raise ValueError('Unknown backbone: {}'.format(args.backbone))
    size_in_bytes = model_size(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print number of parameters
    print(f"Number of parameters: {num_params / (1000.0 ** 2): .3f} M")

    # num_params_gnn = sum(p.numel() for p in model.pyramid_gnn.parameters() if p.requires_grad)
    # print(f"Number of GNN parameters: {num_params_gnn / (1000.0 ** 2): .3f} M")


    print(f"Entire model size: {size_in_bytes / (1024.0 ** 3): .3f} GB")

    print("train",args.train)

    # Initialize a DataFrame to store the training log
    #train_log = pd.DataFrame(columns=["epoch", "train_loss", "train_accuracy"])
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
        if args.model!='swin':
            backbone_params = (
                        list(model.module.conv1.parameters()) +
                        list(model.module.bn1.parameters()) +
                        list(model.module.layer1.parameters()) +
                        list(model.module.layer2.parameters()) +
                        list(model.module.layer3.parameters()) +
                        list(model.module.layer4.parameters()))

            if args.model == "PIGNet_GSPonly_classification":
                last_params = list(model.module.pyramid_gnn.parameters())
                optimizer = optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, backbone_params)},
                    {'params': filter(lambda p: p.requires_grad, last_params)}

                ],lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
            elif args.model == "ASPP":
                last_params = list(model.module.aspp.parameters())
                optimizer = optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, backbone_params)},
                    {'params': filter(lambda p: p.requires_grad, last_params)}

                ],lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

            else:
                last_params = list(model.module.linear.parameters())
                optimizer = optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, backbone_params)},
                    {'params': filter(lambda p: p.requires_grad, last_params)}

                ],
                    lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

        else:# swin
            last_params = list(model.module.parameters())
            optimizer = optim.SGD([
                {'params': filter(lambda p: p.requires_grad, last_params)}

            ], lr=0.001, momentum=0.9, weight_decay=0.0001)

        feature_shape = (2048,33,33)


        collate_fn = partial(make_batch_fn, batch_size=args.batch_size, feature_shape=feature_shape)

        dataset_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.train
        )
        valid_dataset = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False
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
            print("EPOCHS : ", epoch+1," / ",args.epochs)

            loss_sum = 0
            acc_sum = 0
            cnt = 0

            for inputs, target in tqdm(iter(dataset_loader)):
                log = {}
                cur_iter = epoch * len(dataset_loader) + cnt
                lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
                optimizer.param_groups[0]['lr'] = lr
                # if args.model == "deeplab":
                inputs = Variable(inputs.to(args.device))
                target = Variable(target.to(args.device)).long()
                if args.model=='swin':
                    outputs = model(inputs)

                else:
                    outputs,_ = model(inputs)



                outputs_flat = outputs.view(-1, outputs.size(-1))  # 배치 크기 유지하고 마지막 차원을 평탄화

                # 크로스 엔트로피 손실 계산
                loss = F.cross_entropy(outputs_flat, target)
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    pdb.set_trace()
                losses.update(loss.item(), args.batch_size)
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
                    inputs = inputs.to(args.device)
                    labels = torch.tensor(labels).to(args.device)
                    if args.model == 'swin':
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
            # time.sleep(60)
            model.train()

        # train_log.to_csv("training_log.csv", index=False)
    else:
        print("Evaluating !! ")

        torch.cuda.set_device(args.gpu)
        model = model.to(args.device)
        model.eval()
        checkpoint = torch.load(model_fname)

        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
        model.load_state_dict(state_dict)

        valid_dataset = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False
        )

        pixel_similarity_value = [[0 for _ in range(4)] for _ in range(args.n_layer + 1)]
        count = 0
        with torch.no_grad():
            model.eval()
            losses_test = 0.0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(tqdm(valid_dataset)):
                inputs = inputs.to(args.device)
                labels = torch.tensor(labels).to(args.device)
                outputs, layer_outputs = model(inputs)
                _, predicted = outputs.max(1)

                radii=[1,2,4,6]
                plot_cosine_similarity(layer_outputs[0], radii)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()


                #print("gsp_outputs",gsp_outputs)
                # calculate the gsp_outputs output feature map
                # if i % 10 == 0 :
                #     count+=1
                #     result_similarity = []
                #     for idx, out in enumerate(gsp_outputs): #  gsp_output는 8개 layer + 오리지널 데이터 1 개
                #         tmp=compute_similarity(out,labels) #tmp 는 거리에 따른 4개의 output
                #         result_similarity.append(tmp)
                #     #result_similarity 는 8개가 생기는 임시 리스트
                #     for idx, data in enumerate(result_similarity): #result_similarity 의 길이는 8
                #         for idx_, data_ in enumerate(data): # model 는 4개의 길이가 있는
                #             pixel_similarity_value[idx][idx_] += data_




            accuracy = 100 * correct / total
            #for idx, model in enumerate(pixel_similarity_value):
            #    for idx_, data_ in enumerate(model):
            #        pixel_similarity_value[idx][idx_]=data_/count
            print('Accuracy: {:.2f}%'.format(accuracy))
            print("mean of distance ",pixel_similarity_value)

def visualize_sample(img, target):
    """
    Visualizes the image and its corresponding target mask.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].set_title('Image')
    ax[0].axis('off')

    ax[1].imshow(target, cmap='gray')
    ax[1].set_title('Target')
    ax[1].axis('off')

    plt.show()


def get_cosine_similarity(feature_map, center, radius):
    """Calculate the average cosine similarity within a given radius from the center."""
    channels = feature_map.shape[0]
    height, width = feature_map.shape[1:]
    center_y, center_x = center

    similarities = []
    center_vector = feature_map[:, center_y, center_x].reshape(-1)

    for y in range(height):
        for x in range(width):
            if np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2) <= radius:
                vec = feature_map[:, y, x].reshape(-1)
                similarity = 1 - cosine(center_vector.cpu(), vec.cpu())
                similarities.append(similarity)

    return np.mean(similarities) if similarities else 0


def plot_cosine_similarity(feature_map, radii):
    center = (feature_map.shape[1] // 2, feature_map.shape[2] // 2)
    similarities = []

    for radius in radii:
        avg_similarity = get_cosine_similarity(feature_map, center, radius)
        similarities.append(avg_similarity)

    plt.figure(figsize=(10, 6))
    plt.plot(radii, similarities, marker='o')
    plt.xlabel('Radius')
    plt.ylabel('Average Cosine Similarity')
    plt.title('Cosine Similarity vs. Radius')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
