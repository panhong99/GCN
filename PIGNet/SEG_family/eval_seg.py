import argparse
import os
import numpy as np
import torch
import pandas as pd
import pickle
import warnings
import GCN.PIGNet.SEG_family.seg_utils as utils_segmentation
import yaml
import re
import copy

from scipy.io import loadmat
from torch.autograd import Variable
from tqdm.auto import tqdm
from utils import AverageMeter, inter_and_union
from functools import partial
from GCN.PIGNet.SEG_family.seg_dataset import get_dataset
from GCN.PIGNet.SEG_family.seg_models import get_model

warnings.filterwarnings("ignore")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Mode: {config.mode} | Model: {config.model} | Dataset: {config.dataset} | Device: {device}")
    
    dataset = get_dataset(config)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False,
        pin_memory=True, num_workers=config.workers,
        collate_fn=lambda samples: utils_segmentation.make_batch(samples, config.batch_size, feature_shape))

    model_filename = config.infer_params.model_filename
    m_name = re.search(fr"(.*?)_{config.backbone}", model_filename)
    if m_name:
        config.model = m_name.group(1)

    model = get_model(config, dataset)
    model.to(device)
    model.eval()
    model_fname = f'model_{num}/{config.model_number}/segmentation/{config.dataset}/{config.model_type}/{config.model}_{config.backbone}_{config.model_type}_{config.dataset}_{config.n_layer}.pth'

    checkpoint = torch.load(f'/home/hail/pan/GCN/PIGNet/model_{num}/{config.model_number}/segmentation/{config.dataset}/{config.model_type}/{model_filename}'
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

    pred_img = []
    iou_list = []
    img_name = []

    for i in tqdm(range(len(dataset))):
        inputs, target, _, _, _, _ = dataset[i]
        if inputs==None:
            continue

        inputs = Variable(inputs.to(device))
        if config.model == "Mask2Former":
            outputs = model(inputs.unsqueeze(0))
        else:
            outputs, _ = model(inputs.unsqueeze(0))

        _, pred = torch.max(outputs, 1)
        pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
        mask = target.numpy().astype(np.uint8)
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
    parser.add_argument("--mode", type = str, default = "infer", help = "Mode: train or infer")
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