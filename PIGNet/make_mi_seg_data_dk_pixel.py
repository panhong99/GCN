import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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

def mi_seg_data_loader(config, dataset, shuffle=True, sampler=None):
    """DataLoader 생성 함수 - 동일한 셔플 순서 보장"""
    feature_shape = (2048, 33, 33)
    collate_fn = partial(utils_segmentation.make_batch_fn, batch_size=config.batch_size, feature_shape=feature_shape)

    generator = torch.Generator()
    generator.manual_seed(42)

    MI_dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(shuffle and sampler is None),
        pin_memory=True,
        num_workers=config.workers,
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=seed_worker,
        sampler=sampler,
    )

    return MI_dataset_loader

def resize_gt(gt_masks, target_size=33, config_dataset=None):
    """
    빠른 버전 - cv2.resize 사용
    gt_masks: (B, H, W)
    return: (B, target_size, target_size)
    """
    B, H, W = gt_masks.shape
    gt_resized = np.zeros((B, target_size, target_size), dtype=np.uint8)

    for b in range(B):
        resized = cv2.resize(gt_masks[b].astype(np.uint8), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        resized = np.where(resized == 255, 255, resized)
        gt_resized[b] = resized

    return gt_resized


def gather_to_rank0(tensor_cpu, local_rank, world_size, device):
    """
    NCCL all_gather를 통해 모든 rank의 CPU 텐서를 rank 0으로 수집.
    마지막 배치의 크기 불일치를 padding으로 처리.
    Returns list of CPU tensors on rank 0, None on other ranks.
    """
    tensor_gpu = tensor_cpu.to(device)

    # 각 rank의 배치 크기 공유
    local_size = torch.tensor([tensor_gpu.shape[0]], device=device)
    all_sizes = [torch.zeros(1, dtype=local_size.dtype, device=device) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    all_sizes_int = [int(s.item()) for s in all_sizes]
    max_size = max(all_sizes_int)

    # max_size로 padding
    rest_shape = tensor_gpu.shape[1:]
    padded = torch.zeros(max_size, *rest_shape, dtype=tensor_gpu.dtype, device=device)
    padded[:tensor_gpu.shape[0]] = tensor_gpu

    # NCCL은 gather 대신 all_gather 사용
    gathered_gpu = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered_gpu, padded)

    if local_rank == 0:
        result = [g[:s].cpu() for g, s in zip(gathered_gpu, all_sizes_int)]
    else:
        result = None

    del gathered_gpu, padded, tensor_gpu
    return result


def main(config, model_file, model_path, local_rank=0, world_size=1):
    is_distributed = world_size > 1
    is_main = (local_rank == 0)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    dataset = get_dataset(config)
    model = get_model(config, dataset)

    checkpoint = torch.load(os.path.join(model_path, model_file), map_location=device)
    state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}

    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    set_seed(42)
    dataset = get_dataset(config)

    H_size = W_size = 33

    if is_main:
        print(f"\n{'='*60}")
        print(f"Training KMeans models for all layers & pixels (batch-wise)")
        print(f"{'='*60}")
        print(f"Grid size: {H_size} x {W_size}")
        print(f"Total KMeans models: {5 * H_size * W_size} (5 layers × {H_size} × {W_size})")

    # KMeans 모델은 rank 0에서만 관리
    if is_main:
        vq_models = {}
        for layer_idx in range(5):
            for h in range(H_size):
                for w in range(W_size):
                    vq_models[(layer_idx, h, w)] = MiniBatchKMeans(
                        n_clusters=50,
                        random_state=42,
                        n_init=1,
                        batch_size=50,
                        verbose=0
                    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Pass 1: KMeans fitting
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if is_main:
        print(f"\nFitting KMeans models with batch data...")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False) if is_distributed else None
    loader = mi_seg_data_loader(config, dataset, shuffle=False, sampler=sampler)
    loader_list = list(loader)

    for idx in trange(len(loader_list), desc="Training KMeans", disable=not is_main, leave=False):
        inputs, targets = loader_list[idx]

        # 모든 rank가 동일하게 skip하도록 동기화
        if is_distributed:
            valid_flag = torch.tensor([int(inputs is not None)], device=device)
            all_flags = [torch.zeros_like(valid_flag) for _ in range(world_size)]
            dist.all_gather(all_flags, valid_flag)
            if not all(f.item() == 1 for f in all_flags):
                dist.barrier()
                continue
        elif inputs is None:
            continue

        with torch.no_grad():
            inputs_gpu = inputs.to(device)
            outputs, layers_output = model(inputs_gpu)

        if config.model == "Mask2Former":
            layers_output = [layers_output[i] for i in [0, 2, 5, 8, 9]]
            layers_output = [
                torch.nn.functional.interpolate(layer, size=(33, 33), mode='bilinear', align_corners=False)
                for layer in layers_output
            ]

        for layer_idx in range(len(layers_output)):
            layer_cpu = layers_output[layer_idx].cpu()

            if is_distributed:
                gathered = gather_to_rank0(layer_cpu, local_rank, world_size, device)
            else:
                gathered = [layer_cpu]

            if is_main:
                layer_data = torch.cat(gathered, dim=0).numpy()
                B, C, H, W = layer_data.shape

                if config.binary:
                    layer_data = (layer_data > 0).astype(np.float32)

                for h in range(H):
                    for w in range(W):
                        vq_models[(layer_idx, h, w)].partial_fit(layer_data[:, :, h, w])

        del outputs, layers_output, inputs_gpu
        collect()

        if is_distributed:
            dist.barrier()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Pass 2: Prediction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if is_main:
        print(f"\n{'='*60}")
        print(f"Predicting VQ labels for all layers & pixels")
        print(f"{'='*60}")

    gt_masks = []
    pred_masks = []
    all_vq_preds = {layer_idx: [] for layer_idx in range(5)}
    all_vq_valid_masks = {layer_idx: [] for layer_idx in range(5)}

    config.MI = False
    dataset = get_dataset(config)

    infer_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False) if is_distributed else None
    infer_loader = mi_seg_data_loader(config, dataset, shuffle=False, sampler=infer_sampler)
    infer_loader_list = list(infer_loader)

    for idx in trange(len(infer_loader_list), desc="Predicting", disable=not is_main, leave=False):
        inputs, targets = infer_loader_list[idx]

        # 모든 rank가 동일하게 skip하도록 동기화
        if is_distributed:
            valid_flag = torch.tensor([int(inputs is not None)], device=device)
            all_flags = [torch.zeros_like(valid_flag) for _ in range(world_size)]
            dist.all_gather(all_flags, valid_flag)
            if not all(f.item() == 1 for f in all_flags):
                dist.barrier()
                continue
        elif inputs is None:
            continue

        with torch.no_grad():
            inputs_gpu = inputs.to(device)
            outputs, layers_output = model(inputs_gpu)

        if config.model == "Mask2Former":
            layers_output = [layers_output[i] for i in [0, 2, 5, 8, 9]]
            layers_output = [
                torch.nn.functional.interpolate(layer, size=(33, 33), mode='bilinear', align_corners=False)
                for layer in layers_output
            ]

        # GT 수집 (rank 0으로 gather)
        targets_np = targets.numpy().astype(np.uint8)
        if is_distributed:
            targets_t = torch.from_numpy(targets_np.astype(np.float32))
            gathered_targets = gather_to_rank0(targets_t, local_rank, world_size, device)
            if is_main:
                targets_np = torch.cat(gathered_targets, dim=0).numpy().astype(np.uint8)

        if is_main:
            gt_masks.append(resize_gt(targets_np, target_size=33, config_dataset=config.dataset))

        for layer_idx in range(len(layers_output)):
            layer_cpu = layers_output[layer_idx].cpu()

            if is_distributed:
                gathered = gather_to_rank0(layer_cpu, local_rank, world_size, device)
            else:
                gathered = [layer_cpu]

            if is_main:
                layer_data = torch.cat(gathered, dim=0).numpy()
                B, C, H, W = layer_data.shape

                if config.binary:
                    layer_data = (layer_data > 0).astype(np.float32)

                vq_pred_mask = np.zeros((B, H, W), dtype=np.int32)
                for h in range(H):
                    for w in range(W):
                        pixel_features = layer_data[:, :, h, w]
                        vq_pred_mask[:, h, w] = vq_models[(layer_idx, h, w)].predict(pixel_features)

                all_vq_preds[layer_idx].append(vq_pred_mask)

        del outputs, layers_output, inputs_gpu
        collect()

        if is_distributed:
            dist.barrier()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 결과 저장 (rank 0만)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if is_main:
        print(f"\nSaving results...")

        for layer_idx in range(5):
            vq_labels = np.concatenate(all_vq_preds[layer_idx], axis=0)
            with open(config.output_folder + f'/layer_{layer_idx}.pkl', 'wb') as f:
                pickle.dump(vq_labels, f)

        gt_final = np.concatenate(gt_masks, axis=0).astype(np.int32)
        gt_final = np.where(gt_final == 255, -1, gt_final)

        with open(config.output_folder + f'/gt_labels.pkl', 'wb') as f:
            pickle.dump(gt_final, f)

        print(f"Results saved to {config.output_folder}")

        del vq_models, all_vq_preds, gt_masks, pred_masks
        collect()

    if is_distributed:
        dist.barrier()


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

    # DDP 초기화 (torchrun으로 실행 시 LOCAL_RANK 환경변수가 설정됨)
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank >= 0:
        dist.init_process_group(backend='nccl')
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        if local_rank == 0:
            print(f"DDP initialized: {world_size} GPUs")
    else:
        local_rank = 0
        world_size = 1

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

        if model_key in ["PIGNet"]:
            continue

        if local_rank == 0:
            print(f"\n>>> Processing Model: {model_key}")

        for p_key, f_list in process_dict.items():
            for f_val in f_list:
                f_name = zoom_name_map.get(f_val, str(f_val)) if p_key == "zoom" else str(f_val)

                output_folder = os.path.join(
                    "/home/hail/pan/HDD/MI_dataset",
                    "pixel_dataset",
                    config.dataset,
                    config.backbone,
                    config.model_type,
                    model_key,
                    p_key,
                    f_name
                )

                os.makedirs(output_folder, exist_ok=True)

                iter_config = argparse.Namespace(**vars(config))
                iter_config.model = model_key
                iter_config.factor = f_val
                iter_config.factor_name = f_name
                iter_config.infer_params.process_type = p_key
                iter_config.output_folder = output_folder
                iter_config.crop_size = 512 if iter_config.model == "Mask2Former" else 513
                iter_config.batch_size = 50
                main(iter_config, model_file, model_path, local_rank, world_size)

    if dist.is_initialized():
        dist.destroy_process_group()
