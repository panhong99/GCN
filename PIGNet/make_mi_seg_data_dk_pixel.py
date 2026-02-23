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
from make_segmentation_dataset import get_dataset
from make_segmentation_model import get_model
from torch.autograd import Variable
from sklearn.cluster import KMeans
import random
import pickle
import utils_segmentation as utils_segmentation
from gc import collect
from functools import partial
import cv2

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

def main(config, model_file, model_path):

    device = f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu"

    config.batch_size = 16
    dataset = get_dataset(config)
    model = get_model(config, dataset)

    checkpoint = torch.load(os.path.join(model_path, model_file), map_location=device)

    if config.model == "PIGNet_GSPonly":
        layer_num = 5
        state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}

    else: # ASPP
        layer_num = 5
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}

    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    n_clusters = 50
    print(f"KMeans clusters: {n_clusters} (fixed)")
    
    # ===== 레이어 전 공통 데이터셋 로더 생성 =====
    print(f"[INFO] Creating dataset loader...")
    set_seed(42)
    dataset = get_dataset(config)
    MI_dataset_loader = mi_seg_data_loader(config, dataset, shuffle=True)
    
    # 전체 배치에서 모든 layer의 output 수집
    print(f"[PASS 1] Collecting all layer outputs from entire dataset...")
    # all_layer_outputs[layer_idx][(h, w)] = [(feature, sample_idx), ...]
    all_layer_outputs = {i: {} for i in range(5)}
    all_gt_masks = []
    gt_masks_resized = {i: [] for i in range(5)}
    sample_count_per_batch = []
    sample_offset = 0  # 누적 샘플 인덱스
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(MI_dataset_loader, desc="Collecting outputs", leave=False)):
        if inputs is None:
            continue
        
        with torch.no_grad():
            inputs = Variable(inputs.to(device))
            outputs, layers_output = model(inputs)
        
        _, _, H, W = layers_output[0].shape
        targets_np = targets.numpy().astype(np.uint8)
        batch_size = inputs.shape[0]
        
        # GT mask 수집
        all_gt_masks.append(targets_np)
        batch_size = inputs.shape[0]
        sample_count_per_batch.append(batch_size)
        
        # 각 레이어의 output 수집
        for layer_idx in range(5):
            layer_data = layers_output[layer_idx].cpu().numpy()  # (B, C, H, W)
            
            if config.binary == True:
                layer_data = (layer_data > 0).astype(np.float32)
            
            # GT 유효성 마스크 생성
            valid_mask_resized = resize_gt_fast(targets_np, target_size=H)
            gt_mask = (valid_mask_resized != 0)  # (B, H, W)
            
            B, C = layer_data.shape[0], layer_data.shape[1]
            
            # 픽셀 위치별로 feature 수집
            for h in range(H):
                for w in range(W):
                    pixel_key = (h, w)
                    if pixel_key not in all_layer_outputs[layer_idx]:
                        all_layer_outputs[layer_idx][pixel_key] = []
                    
                    # 이 픽셀에 대해 모든 샘플의 feature 수집
                    for b in range(B):
                        if gt_mask[b, h, w]:  # valid한 경우만
                            feature = layer_data[b, :, h, w]  # (C,)
                            sample_idx = sample_offset + b
                            all_layer_outputs[layer_idx][pixel_key].append((feature, sample_idx))
            
            gt_masks_resized[layer_idx].append(gt_mask)
        
        sample_offset += batch_size
        del layers_output, outputs, inputs
        collect()
    
    del MI_dataset_loader
    collect()
    
    print(f"[INFO] Total samples processed: {sum(sample_count_per_batch)}")
    print(f"[INFO] Number of batches: {len(sample_count_per_batch)}")
    
    # ===== PASS 2: 각 레이어별 픽셀별 MiniBatchKMeans 학습 =====
    print(f"\n[PASS 2] Training MiniBatchKMeans for each layer and pixel position...")
    
    total_samples = sum(sample_count_per_batch)
    
    for layer_idx in range(5):
        print(f"\n--- Layer {layer_idx} ---")
        
        # 결과 저장용
        vq_labels_full = -np.ones((total_samples, 33, 33), dtype=np.int32)
        
        num_pixels = len(all_layer_outputs[layer_idx])
        print(f"Training KMeans for {num_pixels} pixel positions...")
        
        for pixel_idx, (pixel_key, features_and_samples) in enumerate(all_layer_outputs[layer_idx].items()):
            if (pixel_idx + 1) % 100 == 0:
                print(f"  Processed {pixel_idx + 1}/{num_pixels} pixels")
            
            h, w = pixel_key
            
            # 이 픽셀의 모든 feature와 샘플 인덱스 추출
            if len(features_and_samples) < n_clusters:
                # feature 개수가 클러스터 수보다 적으면 건너뛰기
                continue
            
            features_list = [item[0] for item in features_and_samples]  # feature만 추출
            sample_indices = [item[1] for item in features_and_samples]  # sample_idx만 추출
            features_array = np.array(features_list)  # (num_samples, C)
            
            # MiniBatchKMeans 학습
            mbkm = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256, n_init=3)
            labels = mbkm.fit_predict(features_array)
            
            # 라벨을 샘플 위치에 매핑
            for label_idx, sample_idx in enumerate(sample_indices):
                vq_labels_full[sample_idx, h, w] = labels[label_idx]
        
        # 저장
        with open(config.output_folder + f'/layer_{layer_idx}.pkl', 'wb') as f:
            pickle.dump(vq_labels_full, f)
            print(f"Saved layer_{layer_idx} with shape {vq_labels_full.shape}")
        
        del all_layer_outputs[layer_idx], gt_masks_resized[layer_idx], vq_labels_full
        collect()
    
    # GT labels 저장
    if all_gt_masks:
        gt_concatenated = np.concatenate(all_gt_masks, axis=0)
        with open(config.output_folder + f'/gt_labels.pkl', 'wb') as f:
            pickle.dump(gt_concatenated, f)
            print(f"\nSaved gt_labels with shape {gt_concatenated.shape}")
    
    del all_gt_masks
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
                    config.dataset, 
                    "pixel_dataset",
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

                main(iter_config, model_file, model_path)
                