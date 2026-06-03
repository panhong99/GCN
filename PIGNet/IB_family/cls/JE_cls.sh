#!/bin/bash
set -e  # 에러 발생 시 즉시 중단

SCRIPT="/home/hail/pan/GCN/PIGNet/JE_cls_main.py"
MODEL="vit"

COUNT=0

if [ "$MODEL" == "vit" ]; then
    BACKBONES="resnet101"
    TOTAL=6
else
    BACKBONES="resnet50 resnet101"
    TOTAL=12
fi

for backbone in $BACKBONES; do
    for dataset in CIFAR-10 CIFAR-100 imagenet; do
        for model_type in scratch pretrained; do
            COUNT=$((COUNT + 1))
            echo ""
            echo "=================================================="
            echo "  [$COUNT/$TOTAL] backbone=$backbone | dataset=$dataset | model_type=$model_type"
            echo "=================================================="

            python3 $SCRIPT \
                --model       $MODEL      \
                --backbone    $backbone   \
                --dataset     $dataset    \
                --model_type  $model_type

            echo "[OK] [$COUNT/$TOTAL] Done"
        done
    done
done

echo ""
echo "All $TOTAL runs completed!"
