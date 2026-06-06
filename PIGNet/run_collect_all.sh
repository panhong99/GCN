#!/bin/bash
# run_collect_all.sh
# Generates segmentation result matrices for all 8 combinations:
#   2 backbones × 2 datasets × 2 model_types = 8
# Each run loads all 3 architectures (PIGNet_GSPonly / ASPP / Mask2Former)
# from their respective model_num folders (1=PIGNet_GSPonly, 2=ASPP, 3=Mask2Former).

set -e

SCRIPT="/home/hail/pan/GCN/PIGNet/collect_img.py"
BACKBONES=("resnet50" "resnet101")
DATASETS=("pascal" "cityscape")
MODEL_TYPES=("scratch" "pretrained")

combo=0
total=8

for backbone in "${BACKBONES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for model_type in "${MODEL_TYPES[@]}"; do
            combo=$((combo + 1))
            echo ""
            echo "============================================================"
            echo " Combination ${combo}/${total}:"
            echo "   backbone=${backbone}  dataset=${dataset}  model_type=${model_type}"
            echo "============================================================"

            python3 "${SCRIPT}" \
                --backbone   "${backbone}" \
                --dataset    "${dataset}" \
                --model_type "${model_type}"

            echo " [Done] ${combo}/${total}"
        done
    done
done

echo ""
echo "All ${total} combinations complete (24 individual model checkpoints evaluated)."
echo "Results saved to: /home/hail/pan/GCN/PIGNet/seg_result_imgs/"
