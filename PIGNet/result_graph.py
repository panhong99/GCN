import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 추출 및 정리 (이미지 92811f.png 기반)
# 성능 저하율 (%) 값만 사용합니다.
data = {
    'Zoom Ratio': ['100%(Base)', '1%', '10%', '25%', '50%', '175%', '275%', '400%'],
    'PigNet-101': [0.00, -96.00, -78.00, -47.00, -23.00, -1.00, -2.00, -9.00],
    'GSP-Only': [0.00, -95.00, -64.00, -35.00, -18.00, 0.00, -2.00, -7.00],
    'ASPP-101': [0.00, -97.00, -64.00, -35.00, -18.00, 0.00, 0.00, -6.00],
    'Mask2Former': [0.00, -88.00, -78.00, -55.00, -27.00, -7.00, -11.00, -19.00]
}

df = pd.DataFrame(data)

# 2. X축 순서 정의 및 데이터프레임 정렬
# X축 순서를 100%를 가장 왼쪽에 고정하고, 그 다음 배율 내림차순으로 설정
ordered_ratios_fixed_100_desc = ['100%(Base)', '400%', '275%', '175%', '50%', '25%', '10%', '1%']
df['Zoom Ratio'] = pd.Categorical(df['Zoom Ratio'], categories=ordered_ratios_fixed_100_desc, ordered=True)
df = df.sort_values('Zoom Ratio').reset_index(drop=True)

# 3. Y축 데이터 처리: 성능 저하율의 절대값 사용 (Degrade Rate %)
# 모든 음수 값은 성능 저하 정도를 나타내기 위해 절대값으로 변환합니다.
for col in df.columns[1:]:
    df[col] = df[col].abs()

# 4. 그래프 그리기
plt.figure(figsize=(12, 6))

for model in df.columns[1:]:
    # X축은 순서대로 정렬된 범주형 레이블을 사용하여 100%(Base)를 가장 왼쪽 기준점으로 만듭니다.
    plt.plot(df['Zoom Ratio'], df[model], marker='o', linewidth=2, label=model)

# 5. 그래프 레이블 및 제목 설정
plt.title('segmentation cityscape Degradation by Zoom Ratio (100% Baseline)', fontsize=16)
plt.xlabel('Zoom Ratio', fontsize=12)
plt.ylabel('Degradation Rate (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Model', loc='upper right')

# 6. 시각적 기준선 설정
# 100%(Base) 지점에 빨간색 수직선 추가
plt.axvline(x='100%(Base)', color='red', linestyle='-', linewidth=2, label='Baseline (100%)')

# Y축 범위 및 눈금 설정 (반전)
max_degrade = df.iloc[:, 1:].max().max()
plt.ylim(max_degrade * 1.05, 0)  # Y축 반전: 높은 값이 아래로
plt.yticks(np.arange(0, 101, 10))

# 7. 그래프 저장
plt.tight_layout()
plt.savefig('zoom_ratio_performance_desc_flipped.png', dpi=300, bbox_inches='tight')
print("Graph saved as 'zoom_ratio_performance_desc_flipped.png'.")