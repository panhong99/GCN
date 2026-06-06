# PIGNet — Experiment Repository

A research codebase for classification and segmentation experiments using PIGNet and baseline models across multiple backbones and datasets, with Information Bottleneck (IB) analysis support.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [IB Analysis](#ib-analysis)
- [Data Storage Structure](#data-storage-structure)

---

## Repository Structure

```
pan/GCN/PIGNet/
├── data/                          # Dataset definition files
│   ├── cifar-10/
│   ├── cifar-100/
│   ├── cityscape/
│   ├── imagenet-100/
│   ├── VOCdevkit/                 # Pascal VOC
│   └── pascal_seg_colormap.mat
│
├── model_src/                     # Model definitions
│   ├── common/
│   ├── cvnets/
│   ├── options/
│   ├── utils/
│   ├── Classification_resnet.py
│   ├── Classification_vit.py
│   ├── PIGNet.py
│   ├── PIGNet_GSPonly.py
│   ├── PIGNet_classification.py
│   ├── PIGNet_GSPonly_classification.py
│   ├── ASPP.py
│   ├── Mask2Former.py
│   ├── swin.py
│   └── plot.py
│
├── Mask2Former_models/            # Mask2Former architecture
│   ├── ops/
│   ├── msdeformattn.py
│   ├── position_encoding.py
│   ├── swin.py
│   └── transformer.py
│
├── IB_family/                     # IB experiment scripts
│   ├── cls/
│   │   ├── IB_cls_figures/        # Output figures (scatter, barplot, etc.)
│   │   ├── VQ_cls.py
│   │   ├── JE_cls_main.py
│   │   ├── JE_calcul_cls.py
│   │   ├── JE_figure_cls.py
│   │   ├── IB_cls.sh              # Run VQ over all model combinations
│   │   ├── JE_cls.sh              # Run full JE/KDE pipeline
│   │   └── config_cls_MI.yaml
│   └── seg/
│       ├── IB_seg_figures/
│       │   ├── ALL_MODELS/
│       │   ├── ASPP/
│       │   ├── Mask2Former/
│       │   └── PIGNet_GSPonly/
│       ├── VQ_seg.py
│       ├── JE_seg_main.py
│       ├── JE_calcul_seg.py
│       ├── JE_figure_seg.py
│       ├── IB_seg.sh
│       ├── JE_seg.sh
│       └── config_seg_MI.yaml
│
├── model_101/                     # Checkpoints — ResNet-101 backbone
│   ├── 1/                         # Model run 1 (PIGNet_GSPonly / PIGNet_GSPonly_cls)
│   │   ├── classification/
│   │   │   ├── CIFAR-10/{pretrained,scratch}/
│   │   │   ├── CIFAR-100/{pretrained,scratch}/
│   │   │   └── imagenet/{pretrained,scratch}/
│   │   └── segmentation/
│   │       ├── cityscape/{pretrained,scratch}/
│   │       └── pascal/{pretrained,scratch}/
│   ├── 2/                         # Model run 2 (ASPP)
│   └── 3/                         # Model run 3 (Mask2Former)
│
├── model_50/                      # Checkpoints — ResNet-50 backbone (same structure)
│
├── cityscapes.py
├── pascal.py
├── cls_dataset.py / seg_dataset.py
├── cls_models.py  / seg_models.py
├── cls_utils.py   / seg_utils.py
├── config_classification.yaml
├── config_segmentation.yaml
├── train_cls.py   / train_seg.py
├── eval_cls.py    / eval_seg.py
└── utils.py
```

> **Model numbering:** 3 independent runs per configuration are used to obtain statistically reliable results.

---

## Configuration

Edit the relevant config file before running any experiment.

| Config file | Task |
|---|---|
| `config_classification.yaml` | Classification |
| `config_segmentation.yaml` | Segmentation |

**Key parameters to set:**

| Parameter | Options |
|---|---|
| `backbone` | `resnet50`, `resnet101` |
| `dataset` (cls) | `CIFAR-10`, `CIFAR-100`, `imagenet` |
| `dataset` (seg) | `pascal`, `cityscape` |
| `model_type` | `scratch`, `pretrained` |

---

## Training

```bash
# Classification
python train_cls.py

# Segmentation
python train_seg.py
```

---

## Evaluation

> Set `model_number` in the config file before running.

```bash
# Classification
python eval_cls.py

# Segmentation
python eval_seg.py
```

---

## IB Analysis

IB analysis runs in two stages: **VQ** (vector quantization) and **JE/KDE** (joint entropy & kernel density estimation).

### Classification

```bash
# Stage 1: VQ for all model combinations
bash IB_family/cls/IB_cls.sh

# Stage 2: JE/KDE computation & figure generation
bash IB_family/cls/JE_cls.sh
```

### Segmentation

```bash
bash IB_family/seg/IB_seg.sh
bash IB_family/seg/JE_seg.sh
```

Output figures are saved to `IB_family/{cls,seg}/IB_{cls,seg}_figures/`.

---

## Data Storage Structure

IB intermediate data (VQ, JE, KDE) is stored on external storage at `pan/HDD/IB_dataset/`.

```
pan/HDD/IB_dataset/
├── CIFAR-10/
│   ├── resnet101/
│   │   ├── pretrained/
│   │   │   ├── PIGNet_GSPonly_classification/
│   │   │   │   └── zoom/
│   │   │   │       └── 1/
│   │   │   │           ├── layer_0.pkl ~ layer_4.pkl
│   │   │   │           ├── gt_labels.pkl
│   │   │   │           ├── analysis_cache_same_diff_joint.pkl  # JE data
│   │   │   │           └── kde_cache_contour.pkl               # KDE data
│   │   │   ├── Resnet/
│   │   │   └── vit/
│   │   └── scratch/
│   │       ├── PIGNet_GSPonly_classification/
│   │       ├── Resnet/
│   │       └── vit/
│   └── resnet50/
│       ├── pretrained/
│       └── scratch/
│
├── CIFAR-100/          # same structure as CIFAR-10
├── imagenet/           # same structure as CIFAR-10
│
├── cityscape/
│   ├── resnet101/
│   │   ├── pretrained/
│   │   │   ├── PIGNet_GSPonly/
│   │   │   │   └── zoom/1/
│   │   │   │       ├── layer_0.pkl ~ layer_4.pkl
│   │   │   │       ├── gt_labels.pkl
│   │   │   │       ├── analysis_cache_same_diff_joint.pkl
│   │   │   │       └── kde_cache_contour.pkl
│   │   │   ├── ASPP/
│   │   │   └── Mask2Former/
│   │   └── scratch/
│   │       ├── PIGNet_GSPonly/
│   │       ├── ASPP/
│   │       └── Mask2Former/
│   └── resnet50/
│       ├── pretrained/
│       └── scratch/
│
└── pascal/             # same structure as cityscape
```
