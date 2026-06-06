#!/bin/bash
# run_collect_all.sh
# Generates segmentation result matrices for all 8 combinations:
#   2 backbones × 2 datasets × 2 model_types = 8
#
# Each run loads all 3 architectures (PIGNet_GSPonly / ASPP / Mask2Former)
# by searching model_{50|101}/1..3/ automatically — no --model_num needed.
#
# Output:
#   seg_result_imgs/matrices/{R50|R101}/{model_type}/{dataset}/{process}_matrix.png
#   appendix/{dataset}/{R50|R101}/{model_type}/GT/   GT_seg/   pred_seg/{arch}/
#   appendix/{dataset}/{R50|R101}/{model_type}/{process}_table.tex

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
echo "All ${total} combinations complete."
echo "Matrices : /home/hail/pan/GCN/PIGNet/seg_result_imgs/matrices/"
echo "Appendix : /home/hail/pan/GCN/PIGNet/appendix/"
