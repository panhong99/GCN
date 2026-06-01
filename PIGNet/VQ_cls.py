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
from cls_dataset import get_dataset
from cls_models import get_model
from torch.autograd import Variable
from sklearn.cluster import KMeans
import random
import pickle
import cls_utils as utils_classification
from gc import collect
from functools import partial
import cv2
from tqdm import trange
import torch.nn.functional as F

warnings.filterwarnings("ignore")

def resize_layers_shape(layers_output, grid_size):
    """
    Layer outputs를 grid_size에 맞춰 자동으로 reshape
    - 크기가 크면: downsampling
    - 크기가 작으면: upsampling
    - 크기가 같으면: 그대로 유지
    
    Args:
        layers_output: List of layer outputs (각 element는 torch.Tensor 또는 numpy array)
                      shape: (B, C, H, W) for each layer
        grid_size: Target grid size (정사각형) - 모든 layer를 이 크기로 변환
    
    Returns:
        resized_layers: List of resized layer outputs with shape (B, C, grid_size, grid_size)
    """
    resized_layers = []
    
    for layer_idx, layer_data in enumerate(layers_output):
        # torch tensor이면 detach해서 GPU에서 CPU로 이동
        if isinstance(layer_data, torch.Tensor):
            layer_tensor = layer_data.detach().cpu()
        else:
            # numpy array면 torch tensor로 변환
            layer_tensor = torch.from_numpy(layer_data).float()
                
        B, C, H, W = layer_tensor.shape
        
        # H와 W가 grid_size와 다르면 interpolate (크기 상관없이 grid_size로 변환)
        if H != grid_size or W != grid_size:
            
            # interpolate 수행
            layer_resized = F.interpolate(
                layer_tensor, 
                size=(grid_size, grid_size),  # ★ 항상 이 크기로 변환 (어떤 입력 크기든 가능)
                mode='bilinear', 
                align_corners=True
            )
            resized_layers.append(layer_resized.detach().cpu().numpy())
        else:
            resized_layers.append(layer_tensor.detach().cpu().numpy())
    
    return resized_layers


def main(config, model_file, model_path):

    if config.dataset != "imagenet":
        grid_size = 8
    else:
        grid_size = 14

    print(f"{config.dataset}")

    device = f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu"
    dataset, _, MI_dataset_loader = get_dataset(config)
    model = get_model(config, dataset)

    checkpoint = torch.load(os.path.join(model_path, model_file), map_location=device)

    # 체크포인트 key가 DDP 사용여부에 따라서 좀 다름 -> 에러로그 보고 변경해주면 됨
    if config.model == "Resnet" or config.model == "vit":
        layer_num = 5
        
        # ★ PIXEL별 KMeans: 각 (layer, h, w)마다 독립적인 모델
        vq_models = {}
        for layer_idx in range(layer_num):
            for h in range(grid_size):
                for w in range(grid_size):
                    key = (layer_idx, h, w)
                    vq_models[key] = MiniBatchKMeans(
                        n_clusters=50,
                        random_state=42,
                        n_init=1,
                        batch_size=50,
                        verbose=0
                    )

    elif config.model == "PIGNet_GSPonly_classification":
        backbone_num, gsp_layer_num = 4, 5  # backbone 4 + GSP block 5

        # ★ PIXEL별 KMeans: Backbone layers
        back_vq_models = {}
        for layer_idx in range(backbone_num):
            for h in range(grid_size):
                for w in range(grid_size):
                    key = (layer_idx, h, w)
                    back_vq_models[key] = MiniBatchKMeans(
                        n_clusters=50,
                        random_state=42,
                        n_init=1,
                        batch_size=50,
                        verbose=0
                    )

        # ★ PIXEL별 KMeans: GSP layers
        gsp_vq_models = {}
        for layer_idx in range(gsp_layer_num):
            for h in range(grid_size):
                for w in range(grid_size):
                    key = (layer_idx, h, w)
                    gsp_vq_models[key] = MiniBatchKMeans(
                        n_clusters=config.cluster_num,
                        random_state=42,
                        n_init=1,
                        batch_size=50,
                        verbose=0
                    )

    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    
    # ===== 레이어 전 공통 데이터셋 로더 생성 =====
    print(f"[INFO] Creating dataset loader...")
    print(f"\n{'='*60}")
    print(f"Training KMeans models for all layers & pixels (batch-wise)")
    print(f"{'='*60}")
    print(f"Total KMeans models: {len(vq_models) if config.model != 'PIGNet_GSPonly_classification' else len(back_vq_models) + len(gsp_vq_models)}")
    print(f"Grid size: {grid_size} x {grid_size}")
            
    for inputs, target in tqdm(iter(MI_dataset_loader), desc="Training KMeans", leave=False):        
        with torch.no_grad():
            inputs = Variable(inputs.to(device))
            target = Variable(target.to(device)).long()
        
        if config.model == "Resnet":
            _, layers_output_ = model(inputs)
            layers_output_.insert(0, inputs.detach().cpu().numpy())
            layers_output = resize_layers_shape(layers_output_, grid_size)

        elif config.model == "vit":
            _, intermidiate = model.forward_intermediates(inputs, indices=[2,5,8,11])
            intermidiate.insert(0, inputs.detach().cpu().numpy())
            layers_output = resize_layers_shape(intermidiate, grid_size)

        elif config.model == "PIGNet_GSPonly_classification":
            _, layers_output_, gsp_layers_output = model(inputs)

            layers_output_.insert(0, inputs.detach().cpu().numpy())
            backbone_layers_output = resize_layers_shape(layers_output_, grid_size)

            # ★ GSP layers도 resize
            gsp_layers_output.insert(0, inputs.detach().cpu().numpy())
            gsp_layers_output_resized = resize_layers_shape(gsp_layers_output, grid_size)

        if config.model != "PIGNet_GSPonly_classification":
            # ★ PIXEL별 처리: 각 픽셀마다 해당 모델에만 fit
            for layer_idx in range(len(layers_output)):
                layer_data = layers_output[layer_idx]
                B, C, H, W = layer_data.shape
                
                # 각 픽셀 (h, w)에 대해
                for h in range(H):
                    for w in range(W):
                        # 해당 픽셀의 배치 데이터만 추출: (B, C)
                        pixel_features = layer_data[:, :, h, w]  # (B, C)
                        
                        # 해당 픽셀 전용 KMeans 모델에만 fit
                        key = (layer_idx, h, w)
                        vq_models[key].partial_fit(pixel_features)

            del layers_output, inputs

        else: # PIGNet_GSPonly_classification
            # ★ Backbone layers: PIXEL별 처리
            for layer_idx in range(backbone_num):
                layer_data = backbone_layers_output[layer_idx]
                B, C, H, W = layer_data.shape
                
                for h in range(H):
                    for w in range(W):
                        pixel_features = layer_data[:, :, h, w]  # (B, C)
                        key = (layer_idx, h, w)
                        back_vq_models[key].partial_fit(pixel_features)

            # ★ GSP layers: PIXEL별 처리
            for layer_idx in range(gsp_layer_num):
                gsp_layer_data = gsp_layers_output_resized[layer_idx]
                B, C, H, W = gsp_layer_data.shape

                for h in range(H):
                    for w in range(W):
                        pixel_features = gsp_layer_data[:, :, h, w]  # (B, C)
                        key = (layer_idx, h, w)
                        gsp_vq_models[key].partial_fit(pixel_features)

            del backbone_layers_output, gsp_layers_output_resized, inputs
       
        collect()
    
    print(f"\n{'='*60}")
    print(f"Predicting VQ labels for all layers & pixels")
    print(f"{'='*60}")
    
    config.MI = False    
    dataset, _, MI_dataset_loader = get_dataset(config)

    if config.model != "PIGNet_GSPonly_classification":
        all_vq_preds = {layer_idx: [] for layer_idx in range(layer_num)}
        all_y = []

    else: # PIGNet_GSPonly_classification
        backbone_vq_preds = {layer_idx: [] for layer_idx in range(backbone_num)}
        backbone_y = []

        gsp_vq_preds = {gsp_layer_idx: [] for gsp_layer_idx in range(gsp_layer_num)}   
        gsp_y = []

    for inputs, target in tqdm(iter(MI_dataset_loader), desc="Predicting KMeans", leave=False):        
        with torch.no_grad():
            inputs = Variable(inputs.to(device))
            target = Variable(target.to(device)).long()
            
        if config.model == "Resnet":
            _, layers_output_ = model(inputs)
            layers_output_.insert(0, inputs.detach().cpu().numpy())
            layers_output = resize_layers_shape(layers_output_, grid_size)

        elif config.model == "vit":
            _, intermidiate = model.forward_intermediates(inputs, indices=[2,5,8,11])
            intermidiate.insert(0, inputs.detach().cpu().numpy())
            layers_output = resize_layers_shape(intermidiate, grid_size)

        elif config.model == "PIGNet_GSPonly_classification":
            _, layers_output_, gsp_layers_output = model(inputs)

            layers_output_.insert(0, inputs.detach().cpu().numpy())
            backbone_layers_output = resize_layers_shape(layers_output_, grid_size)

            # ★ GSP layers도 resize
            gsp_layers_output.insert(0, inputs.detach().cpu().numpy())
            gsp_layers_output_resized = resize_layers_shape(gsp_layers_output, grid_size)

        if config.model != "PIGNet_GSPonly_classification":
            all_y.append(target.detach().cpu().numpy())

        else:
            backbone_y.append(target.detach().cpu().numpy())
            gsp_y.append(target.detach().cpu().numpy())

        # ★ PIXEL별 Prediction: 각 픽셀마다 해당 모델로 predict
        if config.model != "PIGNet_GSPonly_classification":
            for layer_idx in range(len(layers_output)):
                layer_data = layers_output[layer_idx]
                B, C, H, W = layer_data.shape
                
                # 결과를 담을 배열
                if layer_idx not in all_vq_preds or len(all_vq_preds[layer_idx]) == 0:
                    layer_result = np.zeros((B, H, W), dtype=np.int32)
                else:
                    layer_result = np.zeros((B, H, W), dtype=np.int32)
                
                # 각 픽셀마다
                for h in range(H):
                    for w in range(W):
                        pixel_features = layer_data[:, :, h, w]  # (B, C)
                        key = (layer_idx, h, w)
                        
                        # 해당 픽셀 전용 모델로 predict
                        pixel_pred = vq_models[key].predict(pixel_features)  # (B,)
                        layer_result[:, h, w] = pixel_pred
                
                all_vq_preds[layer_idx].append(layer_result)
    
        else: # PIGNet_GSPonly_classification
            # ★ Backbone layers: PIXEL별 Prediction
            for backbone_layer_idx in range(backbone_num):
                backbone_layer_data = backbone_layers_output[backbone_layer_idx]
                B, C, H, W = backbone_layer_data.shape
                
                layer_result = np.zeros((B, H, W), dtype=np.int32)
                
                for h in range(H):
                    for w in range(W):
                        pixel_features = backbone_layer_data[:, :, h, w]  # (B, C)
                        key = (backbone_layer_idx, h, w)
                        pixel_pred = back_vq_models[key].predict(pixel_features)  # (B,)
                        layer_result[:, h, w] = pixel_pred
                
                backbone_vq_preds[backbone_layer_idx].append(layer_result)

            # ★ GSP layers: PIXEL별 Prediction
            for gsp_layer_idx in range(gsp_layer_num):
                gsp_layer_data = gsp_layers_output_resized[gsp_layer_idx]
                B, C, H, W = gsp_layer_data.shape

                layer_result = np.zeros((B, H, W), dtype=np.int32)
                
                for h in range(H):
                    for w in range(W):
                        pixel_features = gsp_layer_data[:, :, h, w]  # (B, C)
                        key = (gsp_layer_idx, h, w)
                        pixel_pred = gsp_vq_models[key].predict(pixel_features)  # (B,)
                        layer_result[:, h, w] = pixel_pred
                
                gsp_vq_preds[gsp_layer_idx].append(layer_result)
    
    # ===== Step 5: 결과 저장 =====
    print(f"\nSaving results...")

    if config.model != "PIGNet_GSPonly_classification":    
        for layer_idx in range(layer_num):        
            vq_labels = np.concatenate(all_vq_preds[layer_idx], axis=0)

            with open(config.output_folder + f'/layer_{layer_idx}_{config.cluster_num}.pkl', 'wb') as f:
                pickle.dump(vq_labels, f)
                
        with open(config.output_folder + f'/y_labels_{config.cluster_num}.pkl', 'wb') as f:
            pickle.dump(np.concatenate(all_y, axis=0), f)

    else: # PIGNet_GSPonly_classification
        for layer_idx in range(backbone_num):
            vq_labels = np.concatenate(backbone_vq_preds[layer_idx], axis=0)

            with open(config.output_folder + f'/backbone_layer_{layer_idx}_{config.cluster_num}.pkl', 'wb') as f:
                pickle.dump(vq_labels, f)

        for gsp_layer_idx in range(gsp_layer_num):
            gsp_vq_labels = np.concatenate(gsp_vq_preds[gsp_layer_idx], axis=0)

            with open(config.output_folder + f'/gsp_layer_{gsp_layer_idx}_{config.cluster_num}.pkl', 'wb') as f:
                pickle.dump(gsp_vq_labels, f)

        with open(config.output_folder + f'/y_labels_{config.cluster_num}.pkl', 'wb') as f:
            pickle.dump(np.concatenate(backbone_y, axis=0), f)
    
    print(f"Results saved to {config.output_folder}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/hail/pan/GCN/PIGNet/config_cls_MI.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cluster_num", type=int, default=50)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        
    def dict_to_namespace(d):
        namespace = argparse.Namespace()
        for k, v in d.items():
            setattr(namespace, k, dict_to_namespace(v) if isinstance(v, dict) else v)
        return namespace
    
    config = dict_to_namespace(config_dict)

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
                    "/home/hail/pan/HDD/IB_dataset", 
                    config.dataset, 
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
                iter_config.batch_size = args.cluster_num
                iter_config.cluster_num = args.cluster_num

                main(iter_config, model_file, model_path)
