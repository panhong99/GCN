#!/bin/bash

CONFIG="/home/hail/pan/GCN/PIGNet/config_segmentation.yaml"
MODEL_NUMBER="${1:-1}"  # 첫 번째 인자로 지정, 기본값 1

BACKBONES=("resnet50" "resnet101")
DATASETS=("pascal" "cityscape")
MODEL_TYPES=("scratch" "pretrained")

echo "Using model_number=${MODEL_NUMBER}"

for backbone in "${BACKBONES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for model_type in "${MODEL_TYPES[@]}"; do
            echo "============================================================"
            echo "backbone=${backbone} | dataset=${dataset} | model_type=${model_type} | model_number=${MODEL_NUMBER}"
            echo "============================================================"

            # yaml의 backbone/dataset/model_type/model_number를 임시로 덮어쓴 config 생성
            TMP_CONFIG=$(mktemp /tmp/config_seg_XXXX.yaml)
            sed \
                -e "s/^backbone:.*/backbone: \"${backbone}\"/" \
                -e "s/^dataset:.*/dataset: \"${dataset}\"/" \
                -e "s/^model_type:.*/model_type: \"${model_type}\"/" \
                -e "s/^model_number:.*/model_number: ${MODEL_NUMBER}/" \
                "$CONFIG" > "$TMP_CONFIG"

            python3 /home/hail/pan/GCN/PIGNet/eval_seg.py --config "$TMP_CONFIG"

            rm -f "$TMP_CONFIG"
        done
    done
done

echo "All 8 combinations done (model_number=${MODEL_NUMBER})."
