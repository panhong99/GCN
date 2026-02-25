import argparse
import os
import torch
import numpy as np
import re
import yaml
import warnings
import pickle
import copy
from tqdm.auto import trange
from sklearn.cluster import MiniBatchKMeans
from make_segmentation_dataset import get_dataset
from make_segmentation_model import get_model
from torch.autograd import Variable
from sklearn.cluster import KMeans
import random
import pickle
import cv2
import utils_segmentation as utils_segmentation
from gc import collect
from functools import partial

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.experimental = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

set_seed(42)

warnings.filterwarnings("ignore")

def mi_seg_data_loader(config, dataset, shuffle=True):
    """DataLoader 생성 함수 - 동일한 셔플 순서 보장"""
    feature_shape = (2048, 33, 33)
    collate_fn = partial(utils_segmentation.make_batch_fn, batch_size=config.batch_size, feature_shape=feature_shape)
    
    # 고정된 seed로 동일한 셔플 순서 보장
    generator = torch.Generator()
    generator.manual_seed(42)
    
    MI_dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=config.workers,
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=seed_worker
    )
    
    return MI_dataset_loader

def resize_gt_fast(gt_masks, target_size=33):
    """
    빠른 버전 - cv2.resize 사용
    gt_masks: (B, H, W)
    return: (B, target_size, target_size)
    """
    B, H, W = gt_masks.shape
    gt_resized = np.zeros((B, target_size, target_size), dtype=np.uint8)
    
    for b in range(B):
        # cv2.resize는 (W, H) 순서로 받음
        resized = cv2.resize(gt_masks[b].astype(np.uint8), (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        # 255는 invalid로 처리 (0으로 변경)
        resized = np.where(resized == 255, 0, resized)
        gt_resized[b] = resized
    
    return gt_resized

def calculate_variance(layer_outputs, valid_mask=None):
    """각 layer의 벡터값들의 분산 계산"""
    # layer_outputs: (num_samples, channels, H, W)
    layer_flat = layer_outputs.reshape(-1, layer_outputs.shape[1])  # (num_samples*H*W, channels)
    
    if valid_mask is not None:
        layer_flat = layer_flat[valid_mask.flatten()]
    
    variance = np.var(layer_flat, axis=0)  # 각 채널별 분산
    mean_variance = np.mean(variance)
    std_variance = np.std(variance)
    
    return mean_variance, std_variance, variance

def main(config, model_file, model_path):

    device = f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu"

    dataset = get_dataset(config)
    model = get_model(config, dataset)

    checkpoint = torch.load(os.path.join(model_path, model_file), map_location=device)

    if config.model == "ASPP":
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    else:
        state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}

    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    set_seed(42)
    dataset = get_dataset(config)
    
    # all_layers_for_variance = [[] for _ in range(5)]
    # all_layers_valid_masks_for_variance = [[] for _ in range(5)]
    
    # loader = mi_seg_data_loader(config, dataset, shuffle=False)
    # loader_list = list(loader)
    
    # for idx in trange(len(loader_list), desc="Collecting data", leave=False):
    #     inputs, targets = loader_list[idx]
    #     if inputs is None:
    #         continue
        
    #     with torch.no_grad():
    #         inputs = Variable(inputs.to(device))
    #         outputs, layers_output, _ = model(inputs)
        
    #     targets_np = targets.numpy().astype(np.uint8)
        
    #     for layer_idx in range(len(layers_output)):
            
    #         layer_data = layers_output[layer_idx].cpu().numpy()
    #         all_layers_for_variance[layer_idx].append(layer_data)
        
    #         _, _, layer_h, layer_w = layer_data.shape
    #         valid_mask_resized = resize_gt_fast(targets_np, target_size=layer_h)
        
    #         valid_mask = (valid_mask_resized != 0)
    #         all_layers_valid_masks_for_variance[layer_idx].append(valid_mask)
        
    #     del outputs, layers_output, inputs
    #     collect()
    
    # # Concatenate for variance calculation
    # for layer_idx in range(5):
    #     all_layers_for_variance[layer_idx] = np.concatenate(all_layers_for_variance[layer_idx], axis=0)
    #     all_layers_valid_masks_for_variance[layer_idx] = np.concatenate(all_layers_valid_masks_for_variance[layer_idx], axis=0)
    
    # # ===== Step 2: Variance 계산 및 최적 bin 결정 =====
    # print(f"\n{'='*60}")
    # print(f"Variance Analysis")
    # print(f"{'='*60}")
    
    # optimal_bins_dict = {}
    # for layer_idx in range(5):
    #     mean_var, std_var, channel_var = calculate_variance(
    #         all_layers_for_variance[layer_idx], 
    #         all_layers_valid_masks_for_variance[layer_idx]
    #     )
        
    #     print(f"\nLayer {layer_idx}:")
    #     print(f"  Mean Variance: {mean_var:.6f}")
    #     print(f"  Std Variance: {std_var:.6f}")
    #     print(f"  Min Channel Variance: {np.min(channel_var):.6f}")
    #     print(f"  Max Channel Variance: {np.max(channel_var):.6f}")
    #     print(f"  Valid Pixel Ratio: {np.sum(all_layers_valid_masks_for_variance[layer_idx]) / all_layers_valid_masks_for_variance[layer_idx].size:.4f}")
    
    # del all_layers_for_variance, all_layers_valid_masks_for_variance
    # collect()
    
    print(f"\n{'='*60}")
    print(f"Training KMeans models for all layers (batch-wise)")
    print(f"{'='*60}")
    
    vq_models = {layer_idx: MiniBatchKMeans(
        n_clusters=50, 
        random_state=42, 
        n_init=1,
        batch_size=50,
        verbose=0
    ) for layer_idx in range(5)}
    
    # 두 번째 pass: 배치별로 바로 fit
    print(f"\nFitting KMeans models with batch data...")
    loader = mi_seg_data_loader(config, dataset, shuffle=False)
    loader_list = list(loader)
    
    for idx in trange(len(loader_list), desc="Training KMeans", leave=False):
        inputs, targets = loader_list[idx]
        if inputs is None:
            continue
        
        with torch.no_grad():
            inputs = Variable(inputs.to(device))
            outputs, layers_output = model(inputs)
        
        targets_np = targets.numpy().astype(np.uint8)
        
        for layer_idx in range(len(layers_output)):
            layer_data = layers_output[layer_idx].cpu().numpy()
            B, C, H, W = layer_data.shape
            
            # Binary 처리: 각 채널값이 0보다 크면 1, 아니면 0
            if config.binary == True:
                layer_data = (layer_data > 0).astype(np.float32)  # (B, C, H, W)

            # GT 유효성 마스크 (segmentation 기반)
            valid_mask_resized = resize_gt_fast(targets_np, target_size=H)
            gt_mask = (valid_mask_resized != 255)  # (B, H, W)
            gt_mask_flat = gt_mask.flatten()
            
            # 이진화된 데이터에서 GT 유효 위치만 선택
            layer_flat = layer_data.transpose(0, 2, 3, 1).reshape(-1, C)
            valid_data = layer_flat[gt_mask_flat]
            
            if len(valid_data) > 0:
                vq_models[layer_idx].partial_fit(valid_data)
        
        del outputs, layers_output, inputs
        collect()
    
    # ===== Step 4: 예측 (한번의 loader로 모든 layer 처리) =====
    print(f"\n{'='*60}")
    print(f"Predicting VQ labels for all layers")
    print(f"{'='*60}")
    
    gt_masks = []
    pred_masks = []
    all_vq_preds = {layer_idx: [] for layer_idx in range(5)}
    all_vq_valid_masks = {layer_idx: [] for layer_idx in range(5)}
    
    config.MI = False
    dataset = get_dataset(config)
    infer_loader = mi_seg_data_loader(config, dataset, shuffle=False)
    infer_loader_list = list(infer_loader)
    
    for idx in trange(len(infer_loader_list), desc="Predicting", leave=False):
        inputs, targets = infer_loader_list[idx]
        if inputs is None:
            continue
        
        with torch.no_grad():
            inputs = Variable(inputs.to(device))
            outputs, layers_output = model(inputs)
        
        targets_np = targets.numpy().astype(np.uint8)
        
        # GT 유효성 마스크 (segmentation 기반) - layer 루프 밖에서 한 번만 계산
        valid_mask_resized = resize_gt_fast(targets_np, target_size=33)
        gt_mask = (valid_mask_resized != 255)  # (B, H, W)
        gt_mask_flat = gt_mask.flatten()
        
        # 모든 layer에 대해 예측 수행
        for layer_idx in range(len(layers_output)):
            layer_data = layers_output[layer_idx].cpu().numpy()  # (batch, C, H, W)
            B, C, H, W = layer_data.shape
            
            # Binary 처리: 각 채널값이 0보다 크면 1, 아니면 0
            if config.binary == True:
                layer_data = (layer_data > 0).astype(np.float32)  # (B, C, H, W)
            
            # 이진화된 데이터로 예측 (유효한 부분만)
            layer_flat = layer_data.transpose(0, 2, 3, 1).reshape(-1, C)
            valid_data = layer_flat[gt_mask_flat]
            vq_pred_valid = vq_models[layer_idx].predict(valid_data)
            
            # 전체 데이터 크기의 배열 생성 후 유효한 부분만 채우기
            # 무효 영역은 -1로 둬서 클러스터 id(0~49)와 구분
            vq_pred_full = np.full(len(layer_flat), -1, dtype=np.int32)
            vq_pred_full[gt_mask_flat] = vq_pred_valid
            
            # 예측 결과를 reshape (0~49 범위, 무효 부분은 -1)
            vq_pred_mask = vq_pred_full.reshape(B, H, W)                        
            all_vq_preds[layer_idx].append(vq_pred_mask)
        
        # GT & Pred 저장 (이미 위에서 valid_mask_resized, gt_mask_flat 계산함)
        mask_flat = valid_mask_resized.reshape(-1)
        mask_filtered_flat = np.full(len(mask_flat), -1, dtype=np.int32)
        mask_filtered_flat[gt_mask_flat] = mask_flat[gt_mask_flat]
        gt_masks.append(mask_filtered_flat.reshape(valid_mask_resized.shape))
        
        del outputs, layers_output, inputs
        collect()
    
    # ===== Step 5: 결과 저장 =====
    print(f"\nSaving results...")
    
    for layer_idx in range(5):
        vq_labels = np.concatenate(all_vq_preds[layer_idx], axis=0)
        
        with open(config.output_folder + f'/layer_{layer_idx}.pkl', 'wb') as f:
            pickle.dump(vq_labels, f)
            
    with open(config.output_folder + f'/gt_labels.pkl', 'wb') as f:
        pickle.dump(np.concatenate(gt_masks, axis=0), f)
    
    print(f"Results saved to {config.output_folder}")
    
    del vq_models, all_vq_preds, gt_masks, pred_masks
    collect()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/hail/pan/GCN/PIGNet/config_seg_MI.yaml")
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
    
    model_path = f"/home/hail/pan/GCN/PIGNet/model_{num}/{config.model_number}/segmentation/{config.dataset}/{config.model_type}"
    model_files = sorted(os.listdir(model_path))

    for model_file in model_files:
        m_name = re.search(fr"(.*?)_{config.backbone}", model_file)
        model_key = m_name.group(1) if m_name else ("Mask2Former" if "Mask2Former" in model_file else "unknown")

        # PIGNet, Mask2Former 제외 처리
        if model_key in ["PIGNet", "Mask2Former"]:
            continue

        print(f"\n>>> Processing Model: {model_key}")
        
        for p_key, f_list in process_dict.items():
            for f_val in f_list:
                f_name = zoom_name_map.get(f_val, str(f_val)) if p_key == "zoom" else str(f_val)
                
                output_folder = os.path.join(
                    "/home/hail/pan/HDD/MI_dataset", 
                    "layer_dataset",
                    config.dataset, 
                    config.backbone, 
                    config.model_type, 
                    model_key, 
                    p_key,
                    f_name
                )

                os.makedirs(output_folder, exist_ok=True)

                # 인퍼런스용 독립 설정
                iter_config = argparse.Namespace(**vars(config))
                iter_config.model = model_key
                iter_config.factor = f_val
                iter_config.factor_name = f_name
                iter_config.infer_params.process_type = p_key
                iter_config.output_folder = output_folder

                main(iter_config, model_file, model_path)
                
