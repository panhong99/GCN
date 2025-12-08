import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 통합 데이터 설정 (Performance Change % 값만 사용, 절댓값으로 변환 예정)
data = {
    'Zoom Ratio': ['100%(Base)', '1%', '10%', '25%', '50%', '175%', '275%', '400%'],
    # ResNet-101 Backbone (Source: 845e79.png)
    'ResNet-101': [0.00, -90.00, -82.00, -73.00, -56.00, -43.00, -55.00, -66.00],
    'GSP-Only101': [0.00, -88.00, -82.00, -71.00, -32.00, -27.00, -41.00, -55.00],
    'PigNet-101': [0.00, -90.00, -83.00, -71.00, -35.00, -33.00, -40.00, -55.00],
    'ViT': [0.00, -82.00, -81.00, -74.00, -69.00, -30.00, -38.00, -45.00],
    # ResNet-50 Backbone (Source: 84d679.png)
    'ResNet-50': [0.00, -87.00, -82.00, -72.00, -54.00, -39.00, -53.00, -64.00],
    'GSP-Only50': [0.00, -87.00, -84.00, -71.00, -36.00, -30.00, -46.00, -51.00],
    'PigNet-50': [0.00, -89.00, -85.00, -70.00, -38.00, -28.00, -42.00, -56.00]
}

df = pd.DataFrame(data)

# 2. X축 순서 정의 및 데이터프레임 정렬
# X축 순서를 100%를 가장 왼쪽에 고정하고, 그 다음 배율 내림차순으로 설정
ordered_ratios_fixed_100_desc = ['100%(Base)', '400%', '275%', '175%', '50%', '25%', '10%', '1%']
df['Zoom Ratio'] = pd.Categorical(df['Zoom Ratio'], categories=ordered_ratios_fixed_100_desc, ordered=True)
df = df.sort_values('Zoom Ratio').reset_index(drop=True)

# 3. Y축 데이터 처리: 성능 저하율의 절대값 사용
for col in df.columns[1:]:
    df[col] = df[col].abs()

# 4. 그래프 그리기
plt.figure(figsize=(14, 8))

# 모델별 색상과 선 스타일 정의
model_styles = {
    'ResNet-101': {'color': 'orange', 'linestyle': '-'},
    'GSP-Only101': {'color': 'green', 'linestyle': '-'},
    'PigNet-101': {'color': 'blue', 'linestyle': '-'},
    'ViT': {'color': 'purple', 'linestyle': '-'},
    'ResNet-50': {'color': 'orange', 'linestyle': '--'},
    'GSP-Only50': {'color': 'green', 'linestyle': '--'},
    'PigNet-50': {'color': 'blue', 'linestyle': '--'}
}

# 모든 모델을 순회하며 그래프 그리기
for model in df.columns[1:]:
    style = model_styles.get(model, {'color': 'black', 'linestyle': '-'})
    plt.plot(df['Zoom Ratio'], df[model], marker='o', linewidth=2, 
             color=style['color'], linestyle=style['linestyle'], label=model)

# 5. 그래프 레이블 및 제목 설정
plt.title('clssification cifar-10 pretrained Model Performance Degradation by Zoom Ratio (100% Baseline)', fontsize=16)
plt.xlabel('Zoom Ratio', fontsize=12)
plt.ylabel('Degradation Rate (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Model', loc='upper right', bbox_to_anchor=(1.2, 1.0))

# 6. 시각적 기준선 및 범위 설정
# 100%(Base) 지점에 수직 기준선 추가
plt.axvline(x='100%(Base)', color='red', linestyle='-', linewidth=2, label='Baseline (100%)')

# Y축 범위 및 눈금 설정 (반전)
max_degrade = df.iloc[:, 1:].max().max()
plt.ylim(max_degrade * 1.05, 0)  # Y축 반전: 높은 값이 아래로
plt.yticks(np.arange(0, 101, 10))

# 7. 그래프 저장
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig('integrated_performance_cls.png', dpi=300, bbox_inches='tight')
print("Graph saved as 'integrated_performance_cls.png'.")