import argparse
import os
import torch
import numpy as np
import re
import yaml
import warnings
import pickle
import copy
from tqdm.auto import tqdm
from sklearn.cluster import MiniBatchKMeans
from make_classification_dataset import get_dataset
from make_classification_model import get_model
from torch.autograd import Variable
from sklearn.cluster import KMeans
import random
import pickle
import utils_classification as utils_classification
from gc import collect
from functools import partial
import cv2
from tqdm import trange
import torch.nn.functional as F

warnings.filterwarnings("ignore")

def main(config, model_file, model_path):

    if config.dataset != "imagenet":
        grid_size = 8
    else:
        grid_size = 14

    device = f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu"
    dataset, _, MI_dataset_loader = get_dataset(config)
    model = get_model(config, dataset)

    checkpoint = torch.load(os.path.join(model_path, model_file), map_location=device)

    # 체크포인트 key가 DDP 사용여부에 따라서 좀 다름 -> 에러로그 보고 변경해주면 됨
    if config.model == "Resnet" or config.model == "vit":
        layer_num = 4
    else: # PIGNet_GSPonly_classification
        layer_num = 8 # backbone 4 + GSP block 4

    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    
    # ===== 레이어 전 공통 데이터셋 로더 생성 =====
    print(f"[INFO] Creating dataset loader...")
    print(f"\n{'='*60}")
    print(f"Training KMeans models for all layers (batch-wise)")
    print(f"{'='*60}")
    
    vq_models = {layer_idx: MiniBatchKMeans(
        n_clusters=50, 
        random_state=42, 
        n_init=1,
        batch_size=50,
        verbose=0
    ) for layer_idx in range(layer_num)}
            
    for inputs, target in tqdm(iter(MI_dataset_loader), desc="Training KMeans", leave=False):        
        with torch.no_grad():
            inputs = Variable(inputs.to(device))
            target = Variable(target.to(device)).long()
        
        if config.model == "Resnet":
            outputs, layers_output = model(inputs)

        elif config.model == "vit":
            _, intermidiate = model.forward_intermidiate(inputs, indices=[2,5,8,11])
            layers_output = []
            for i in intermidiate:    
                i_ = F.interpolate(i, size=(grid_size, grid_size), mode='bilinear', align_corners=True)
                layers_output.append(i_)

        else: # PIGNet_GSPonly_classification
            outputs, layers_output = model(inputs)
            
        # layers_output return할 때 최소한 detach는 되야함 
        # detach 안되있으면 근데 return도 안될 듯 -> check해보기
        for layer_idx in range(len(layers_output)):
            layer_data = layers_output[layer_idx].cpu().numpy()
            B, C, H, W = layer_data.shape
                        
            layer_flat = layer_data.transpose(0, 2, 3, 1).reshape(-1, C)
            vq_models[layer_idx].partial_fit(layer_flat)
        
        del outputs, layers_output, inputs
        collect()
    
    print(f"\n{'='*60}")
    print(f"Predicting VQ labels for all layers")
    print(f"{'='*60}")
    
    config.MI = False    
    dataset, _, MI_dataset_loader = get_dataset(config)

    all_vq_preds = {layer_idx: [] for layer_idx in range(layer_num)}
    all_y = []

    for inputs, target in tqdm(iter(MI_dataset_loader), desc="Predicting KMeans", leave=False):        
        with torch.no_grad():
            inputs = Variable(inputs.to(device))
            target = Variable(target.to(device)).long()
            
        if config.model == "Resnet":
            outputs, layers_output = model(inputs)

        elif config.model == "vit":
            _, intermidiate = model.forward_intermidiate(inputs, indices=[2,5,8,11])
            layers_output = []
            for i in intermidiate:    
                i_ = F.interpolate(i, size=(grid_size, grid_size), mode='bilinear', align_corners=True)
                layers_output.append(i_)

        else: # PIGNet_GSPonly_classification
            outputs , layers_output = model(inputs)

        all_y.append(target.detach().cpu().numpy())
            
         # 모든 layer에 대해 예측 수행
        for layer_idx in range(len(layers_output)):
            layer_data = layers_output[layer_idx].cpu().numpy()  # (batch, C, H, W)
            B, C, H, W = layer_data.shape
           
            layer_flat = layer_data.transpose(0, 2, 3, 1).reshape(-1, C)
            vq_pred = vq_models[layer_idx].predict(layer_flat)
            all_vq_preds[layer_idx].append(vq_pred.reshape(B,H,W))
    
    # ===== Step 5: 결과 저장 =====
    # all_vq_preds의 shape을 봐야함 -> 3차원으로 형성이 되있어야함 -> all_batch, H, W 이렇게
    print(f"\nSaving results...")
    
    for layer_idx in range(layer_num):        
        vq_labels = np.concatenate(all_vq_preds[layer_idx], axis=0)

        with open(config.output_folder + f'/layer_{layer_idx}.pkl', 'wb') as f:
            pickle.dump(vq_labels, f)
            
    with open(config.output_folder + f'/y_labels.pkl', 'wb') as f:
        pickle.dump(np.concatenate(all_y, axis=0), f)
    
    print(f"Results saved to {config.output_folder}")
    
    del vq_models, all_vq_preds, all_y

    collect()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/hail/pan/GCN/PIGNet/config_cls_MI.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        
    def dict_to_namespace(d):
        namespace = argparse.Namespace()
        for k, v in d.items():
            setattr(namespace, k, dict_to_namespace(v) if isinstance(v, dict) else v)
        return namespace
    
    config = dict_to_namespace(config_dict)

    # 가공 조건 및 이름 매핑
    # process_dict = {
    #     "zoom": [0.1, np.sqrt(0.1), 0.5, np.sqrt(0.5), 1, 1.5, np.sqrt(2.75), 2],
    #     "overlap": [0, 0.1, 0.2, 0.3, 0.5],
    #     "repeat": [1, 3, 6, 9, 12]
    # }

    process_dict = {
        "zoom": [1],
    }

    zoom_name_map = {0.1: "0.1", np.sqrt(0.1): "0.3", 0.5: "0.5", np.sqrt(0.5): "0.7", 
                     1: "1", 1.5: "1.5", np.sqrt(2.75): "1.75", 2: "2.0"}
    
    if config.backbone == "resnet50":
        num=50
    elif config.backbone == "resnet101":
        num=101
    
    model_path = f"/home/hail/pan/GCN/PIGNet/model_{num}/{config.model_number}/classification/{config.dataset}/{config.model_type}"
    model_files = sorted(os.listdir(model_path))

    for model_file in model_files:
        m_name = re.search(fr"classification_(.*?)_{config.backbone}", model_file)
        model_key = m_name.group(1) if m_name else ("vit" if "vit" in model_file else "unknown")

        # PIGNet, Mask2Former 제외 처리
        if model_key in ["PIGNet_classification"]:
            continue

        print(f"\n>>> Processing Model: {model_key}")
        
        for p_key, f_list in process_dict.items():
            for f_val in f_list:
                f_name = zoom_name_map.get(f_val, str(f_val)) if p_key == "zoom" else str(f_val)
                
                output_folder = os.path.join(
                    "/home/hail/pan/HDD/MI_dataset", 
                    config.dataset, 
                    "layer_dataset",
                    config.backbone, 
                    config.model_type, 
                    model_key, 
                    p_key,
                    f_name,
                )

                os.makedirs(output_folder, exist_ok=True)

                # 인퍼런스용 독립 설정
                iter_config = argparse.Namespace(**vars(config))
                iter_config.model = model_key
                iter_config.factor = f_val
                iter_config.factor_name = f_name
                iter_config.infer_params.process_type = p_key
                iter_config.output_folder = output_folder
                iter_config.crop_size= 512 if iter_config.model == "vit" else 513
                main(iter_config, model_file, model_path)
                