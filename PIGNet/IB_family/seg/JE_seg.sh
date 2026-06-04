#!/bin/bash
set -e  # 에러 발생 시 즉시 중단

SCRIPT="/home/hail/pan/GCN/PIGNet/IB_family/seg/JE_seg_main.py"
MODEL="PIGNet_GSPonly"

TOTAL=8
COUNT=0

for backbone in resnet50 resnet101; do
    for dataset in pascal cityscape; do
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

            echo "✓ [$COUNT/$TOTAL] Done"
        done
    done
done

echo ""
echo "All $TOTAL runs completed!"
