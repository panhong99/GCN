#!/bin/bash
set -e  # 에러 발생 시 즉시 중단

CONFIG="/home/hail/pan/GCN/PIGNet/config_seg_MI.yaml"
SCRIPT="/home/hail/pan/GCN/PIGNet/VQ_seg.py"

# config yaml의 특정 필드 덮어쓰기
update_config() {
    python3 -c "
import yaml
with open('$CONFIG', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['backbone']   = '$1'
cfg['dataset']    = '$2'
cfg['model_type'] = '$3'
with open('$CONFIG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"
}

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

            update_config $backbone $dataset $model_type
            python3 $SCRIPT --config $CONFIG

            echo "✓ [$COUNT/$TOTAL] Done"
        done
    done
done

echo ""
echo "All $TOTAL runs completed!"
