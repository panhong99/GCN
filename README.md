# 실험 디렉토리 구조
```
pan/GCN/PIGNet/ 
├── data/  # 데이터 정의 파일
│   ├── cifar-10/
│   ├── cifar-100/
│   ├── cityscape/
│   ├── imagenet-100/
│   ├── VOCdevkit/ # Pascal
│   └── pascal_seg_colormap.mat
│
├── IB_family/ # IB 실험용 폴더
│   ├── cls/
│   │   ├── IB_cls_figures/ # scatter, barplot등 이미지 저장폴더
│   │   ├── VQ_cls.py # VQ 진행
│   │   ├── JE_cls_main.py # JE 연산 및 이미지 생성
│   │   ├── JE_calcul_cls.py # JE, KDE 연산 
│   │   ├── JE_figure_cls.py # 이미지 생성
│   │   ├── IB_cls.sh # VQ실험을 전체 모델 조합(Backbone, model train type)에 대한 실행 파일
│   │   ├── JE_cls.sh # JE 전체실험(JE, KDE 연산 및 이미지 생성)실행 파일
│   │   └── config_cls_MI.yaml # VQ실험에 사용되는 config정의
│   └── seg/
│       ├── IB_seg_figures/
│       │   ├── ALL_MODELS/ # bar plot
│       │   ├── ASPP/ # scatter plot
│       │   ├── Mask2Former/ 
│       │   └── PIGNet_GSPonly/ 
│       ├── VQ_seg.py # 위 cls파일들 정의와 동일
│       ├── JE_seg_main.py
│       ├── JE_calcul_seg.py
│       ├── JE_figure_seg.py
│       ├── IB_seg.sh
│       ├── JE_seg.sh
│       └── config_seg_MI.yaml
│
├── Mask2Former_models/ # Mask2Former model정의 폴더
│   ├── ops/
│   ├── __init__.py
│   ├── msdeformattn.py
│   ├── position_encoding.py
│   ├── swin.py
│   └── transformer.py
│
├── model_src/ # 전체 모델 정의 폴더
│   ├── common/
│   ├── cvnets/
│   ├── options/
│   ├── utils/
│   ├── ASPP.py
│   ├── Classification_resnet.py
│   ├── Classification_vit.py
│   ├── Mask2Former.py
│   ├── PIGNet.py
│   ├── PIGNet_GSPonly.py
│   ├── PIGNet_classification.py
│   ├── PIGNet_GSPonly_classification.py
│   ├── plot.py
│   └── swin.py
│
├── model_101/ # ckpt 저장 폴더(Resnet101 backbone)
│   ├── 1/ # 모델 번호(통계적 결과를 얻기 위한 3개 모델 생성)
│   │   ├── classification/ # PIGNet_GSPonly_classification 모델
│   │   │   ├── CIFAR-10/
│   │   │   │   ├── pretrained/
│   │   │   │   └── scratch/
│   │   │   ├── CIFAR-100/
│   │   │   │   ├── pretrained/
│   │   │   │   └── scratch/
│   │   │   └── imagenet/
│   │   │       ├── pretrained/
│   │   │       └── scratch/
│   │   └── segmentation/ # PIGNet_GSPonly 모델
│   │       ├── cityscape/
│   │       │   ├── pretrained/
│   │       │   └── scratch/
│   │       └── pascal/
│   │           ├── pretrained/
│   │           └── scratch/
│   ├── 2/              # 1과 동일한 구조 (ASPP 모델)
│   └── 3/              # 1과 동일한 구조 (Mask2Former 모델)
│
├── model_50/           # model_101과 동일한 구조 (resnet50 기반)
│   ├── 1/
│   ├── 2/
│   └── 3/
│
├── cityscapes.py # cityscape dataset 정의
├── cls_dataset.py # cls dataset 정의
├── cls_models.py # cls model 정의
├── cls_utils.py # cls dataset 및 model에 대한 utils
├── config_classification.yaml # cls train, eval의 config정의
├── config_segmentation.yaml # seg train, eval의 config정의
├── eval_cls.py # cls eval 
├── eval_seg.py # seg eval
├── pascal.py # pascal dataset 정의
├── seg_dataset.py # seg dataset 정의
├── seg_models.py # seg model 정의
├── seg_utils.py # seg dataset 및 model에 대한 utils
├── train_cls.py # cls model train
├── train_seg.py # seg model train
└── utils.py # 전체 utils
```

# IB 데이터 저장경로 구조
```
pan/HDD/IB_dataset/                  # VQ 및 JE, KDE 데이터 저장 경로 (외부 저장소)
├── CIFAR-10/                        # cls 데이터셋
│   ├── resnet101/
│   │   ├── pretrained/
│   │   │   ├── PIGNet_GSPonly_classification/
│   │   │   │   └── zoom/
│   │   │   │       └── 1/
│   │   │   │           ├── analysis_cache_same_diff_joint.pkl  # JE 데이터
│   │   │   │           ├── gt_labels.pkl # VQ gt label
│   │   │   │           ├── kde_cache_contour.pkl # KDE 데이터
│   │   │   │           ├── layer_0.pkl
│   │   │   │           ├── layer_1.pkl
│   │   │   │           ├── layer_2.pkl
│   │   │   │           ├── layer_3.pkl
│   │   │   │           └── layer_4.pkl
│   │   │   ├── Resnet/             # 동일한 구조
│   │   │   └── vit/                # 동일한 구조
│   │   └── scratch/
│   │       ├── PIGNet_GSPonly_classification/  # 동일한 구조
│   │       ├── Resnet/
│   │       └── vit/
│   └── resnet50/                   # resnet101과 동일한 구조
│       ├── pretrained/
│       └── scratch/
├── CIFAR-100/                       # CIFAR-10과 동일한 구조
├── imagenet/                        # CIFAR-10과 동일한 구조
├── cityscape/                       # seg 데이터셋
│   ├── resnet101/
│   │   ├── pretrained/
│   │   │   ├── PIGNet_GSPonly/
│   │   │   │   └── zoom/
│   │   │   │       └── 1/
│   │   │   │           ├── analysis_cache_same_diff_joint.pkl
│   │   │   │           ├── gt_labels.pkl
│   │   │   │           ├── kde_cache_contour.pkl
│   │   │   │           ├── layer_0.pkl
│   │   │   │           ├── layer_1.pkl
│   │   │   │           ├── layer_2.pkl
│   │   │   │           ├── layer_3.pkl
│   │   │   │           └── layer_4.pkl
│   │   │   ├── ASPP/               # 동일한 구조
│   │   │   └── Mask2Former/        # 동일한 구조
│   │   └── scratch/
│   │       ├── PIGNet_GSPonly/     # 동일한 구조
│   │       ├── ASPP/
│   │       └── Mask2Former/
│   └── resnet50/                   # resnet101과 동일한 구조
│       ├── pretrained/
│       └── scratch/
└── pascal/                          # cityscape와 동일한 구조
```
