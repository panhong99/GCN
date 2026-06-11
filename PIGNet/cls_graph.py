import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'regular'

# ── 데이터 범위 설정 ──────────────────────────────────────────────────────────
# DATASETS : 열(column) 구성 — 파일 이름 및 subplot 제목에 직접 사용됨
# TASKS    : 행(row) 구성 — 각 행이 하나의 task에 대응
# MODEL_TYPES : 별도 figure로 출력; 파일명 및 Excel sheet 이름에 사용됨
DATASETS         = ['CIFAR-10', 'CIFAR-100', 'ImageNet']
DATASET_DISPLAY  = {'CIFAR-10': 'CIFAR-10', 'CIFAR-100': 'CIFAR-100', 'ImageNet': 'ImageNet-100K'}
TASKS       = ['zoom', 'rotate']
MODEL_TYPES = ['scratch', 'pretrain']

# ── 시각화 스타일 ─────────────────────────────────────────────────────────────
# COLOR_MAP    : 모델별 선/밴드 색상
# LEGEND_ORDER : 범례 표시 순서
# ZORDER_MAP   : 겹칠 때 위에 그려질 모델 우선순위 (값이 클수록 앞)
# DISPLAY_NAME : 범례에 실제로 표시되는 이름
COLOR_MAP    = {'PIGNet_GSPonly': '#D81B60', 'ResNet': '#5C6BC0', 'ViT': '#FF7043'}
LEGEND_ORDER = ['PIGNet_GSPonly', 'ResNet', 'ViT']
ZORDER_MAP   = {'ResNet': 7, 'PIGNet_GSPonly': 10, 'ViT': 4}
DISPLAY_NAME = {'PIGNet_GSPonly': 'PIGNet_GSPOnly', 'ResNet': 'ResNet', 'ViT': 'ViT'}

# x축 라벨 텍스트 (task별)
TASK_XLABEL = {'zoom': 'Zoom Ratio', 'rotate': 'Rotate Ratio'}

# ── 폰트 크기 ────────────────────────────────────────────────────────────────
# 아래 값을 줄이면 해당 요소 텍스트가 작아짐
FS_TITLE  = 33   # subplot 상단 제목 (CIFAR-10 / CIFAR-100 / ImageNet)
FS_LABEL  = 25   # x·y축 라벨
FS_TICK   = 20   # x·y축 눈금 숫자
FS_LEGEND = 33   # 범례 텍스트


# ── 헬퍼 함수 ────────────────────────────────────────────────────────────────

def normalize_model_name(name):
    """Excel 시트의 원본 모델명 → 내부 키(COLOR_MAP 등)로 변환"""
    m = str(name).lower()
    if 'resnet' in m: return 'ResNet'
    if 'gsp'    in m: return 'PIGNet_GSPonly'
    if 'vit'    in m: return 'ViT'
    return str(name)


def extract_ratios(df, task):
    """첫 번째 행(헤더)에서 x축 값(ratio 목록)을 추출. zoom은 소수→% 변환."""
    raw = df.iloc[0, 1:].values
    out = []
    for val in raw:
        if pd.isna(val):
            continue
        if isinstance(val, (float, int)):
            if task.lower() == 'zoom' and 0 < val <= 5.0:
                out.append(float(val * 100))
            else:
                out.append(float(val))
        else:
            m = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
            if m:
                out.append(float(m.group()))
    return np.array(out)


def extract_models(df):
    """
    첫 번째 열에서 모델명 목록을 순서대로 추출.
    ratio/task명·'mean'·'std'·'$'로 시작하는 행은 제외.
    """
    skip = {'ratio', 'zoom', 'rotate', 'overlap', 'mean', 'std'}
    seen, out = set(), []
    for m in df.iloc[:, 0].dropna().astype(str):
        clean = m.strip()
        if clean.lower() not in skip and not clean.startswith('$') and clean not in seen:
            out.append(clean)
            seen.add(clean)
    return out


def extract_model_data(df, models, ratios):
    """
    시트 구조: 행1~N = 각 모델 mean, 행(N+1)~(2N) = 각 모델 std
    반환: {모델명: {ratio값: (mean, std)}}
    """
    data = {}
    for i, model in enumerate(models):
        row_idx = i + 1
        if row_idx >= len(df):
            continue
        means = pd.to_numeric(df.iloc[row_idx,             1:1+len(ratios)], errors='coerce').values
        stds  = pd.to_numeric(df.iloc[row_idx+len(models), 1:1+len(ratios)], errors='coerce').values
        d = {}
        for r, mn, sd in zip(ratios, means, stds):
            rv = float(pd.to_numeric(r,  errors='coerce'))
            mv = float(pd.to_numeric(mn, errors='coerce'))
            sv = float(pd.to_numeric(sd, errors='coerce'))
            if not any(np.isnan([rv, mv, sv])) and rv not in d:
                d[rv] = (mv, sd)
        data[str(model)] = d
    return data


def extract_vit_data(df, task):
    """
    ViT 전용 파서. 시트 구조: row0=ratios, 'mean' 행=평균, 'std' 행=표준편차.
    반환: ({ratio: (mean, std)}, ratios 배열)
    """
    ratios = extract_ratios(df, task)
    first_col = df.iloc[:, 0].astype(str).str.strip().str.lower()
    mean_row = df[first_col == 'mean']
    std_row  = df[first_col == 'std']
    if mean_row.empty or std_row.empty:
        return {}, ratios
    means = pd.to_numeric(mean_row.iloc[0, 1:1+len(ratios)], errors='coerce').values
    stds  = pd.to_numeric(std_row.iloc[0,  1:1+len(ratios)], errors='coerce').values
    d = {}
    for r, mn, sd in zip(ratios, means, stds):
        rv = float(pd.to_numeric(r,  errors='coerce'))
        mv = float(pd.to_numeric(mn, errors='coerce'))
        sv = float(pd.to_numeric(sd, errors='coerce'))
        if not any(np.isnan([rv, mv, sv])) and rv not in d:
            d[rv] = (mv, sv)
    return d, ratios


# ── subplot 그리기 ────────────────────────────────────────────────────────────

def draw_subplot(ax, dataset, task, model_type, is_left_col):
    excel_path = f'/home/hail/pan/GCN/PIGNet/csvs/{dataset}_{model_type}.xlsx'

    try:
        df_101 = pd.read_excel(excel_path, sheet_name=f'{task}_101', header=None)
        df_50  = pd.read_excel(excel_path, sheet_name=f'{task}_50',  header=None)
    except Exception as e:
        ax.text(0.5, 0.5, f'No data\n{e}', ha='center', va='center',
                transform=ax.transAxes, fontsize=FS_LABEL)
        return

    ratios_101 = extract_ratios(df_101, task)
    ratios_50  = extract_ratios(df_50,  task)
    models_101 = extract_models(df_101)
    models_50  = extract_models(df_50)

    data_101 = extract_model_data(df_101, models_101, ratios_101)
    data_50  = extract_model_data(df_50,  models_50,  ratios_50)

    # 두 backbone의 ratio를 합쳐 공통 x축 구성
    all_ratios = sorted(set(ratios_101) | set(ratios_50))
    full_x_idx = np.arange(len(all_ratios))
    full_r2i   = {r: i for i, r in enumerate(all_ratios)}

    # LEGEND_ORDER 기준으로 모델 정렬 → 중요 모델이 마지막에 그려져 앞에 표시됨
    all_models = sorted(
        set(models_101) | set(models_50),
        key=lambda x: LEGEND_ORDER.index(normalize_model_name(x))
                      if normalize_model_name(x) in LEGEND_ORDER else 99
    )
    vit_plotted = False

    # R101=실선, R50=점선으로 구분; ViT는 backbone 없이 한 번만 그림
    for data_dict, models, backbone in [(data_101, models_101, '101'),
                                         (data_50,  models_50,  '50')]:
        linestyle = '-' if backbone == '101' else '--'
        for model_orig in all_models:
            if model_orig not in models:
                continue
            model_name = normalize_model_name(model_orig)
            if model_name == 'ViT':
                if vit_plotted:
                    continue
                linestyle = '-'
                vit_plotted = True
            ratio_data = data_dict.get(model_orig, {})

            px, py, ps = [], [], []
            for r in all_ratios:
                if r in ratio_data:
                    px.append(full_r2i[r])
                    py.append(ratio_data[r][0])
                    ps.append(ratio_data[r][1])

            if not py:
                continue

            color  = COLOR_MAP.get(model_name, 'black')
            zorder = ZORDER_MAP.get(model_name, 3)
            py, ps = np.array(py), np.array(ps)

            ax.plot(px, py, marker='o', color=color, linestyle=linestyle,
                    linewidth=2, markersize=4, zorder=zorder)
            ax.fill_between(px, py - ps, py + ps, color=color, alpha=0.15, zorder=zorder-1)

    # x축 눈금
    ax.set_xticks(full_x_idx)
    if task == 'zoom':
        tick_labels = [f"{int(r)}%" for r in all_ratios]
    else:
        tick_labels = [str(int(r) if r == int(r) else r) for r in all_ratios]
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', rotation_mode='anchor',
                       fontsize=FS_TICK, fontfamily='Times New Roman')
    ax.set_xlabel(TASK_XLABEL[task], fontsize=FS_LABEL, fontfamily='Times New Roman', labelpad=12)

    # y축: 왼쪽 열만 라벨 표시, 항상 0에서 시작
    if is_left_col:
        ax.set_ylabel('Accuracy (%)', fontsize=FS_LABEL, fontfamily='Times New Roman', labelpad=12)
    else:
        ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=FS_TICK, labelleft=True)
    ax.set_ylim(0, 90)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))  # n 단위마다 눈금

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)


def make_legend_handles():
    """
    범례 핸들 생성. ncol=3이면 matplotlib은 열 우선(column-first)으로 채움:
        PIGNet(R101)  ResNet(R101)  ViT
        PIGNet(R50)   ResNet(R50)   [empty]
    → 핸들 순서: PIGNet_R101, PIGNet_R50, ResNet_R101, ResNet_R50, ViT
    ViT는 backbone 구분 없이 solid 한 줄만.
    """
    handles = []
    for model_name in ['PIGNet_GSPonly', 'ResNet']:
        display = DISPLAY_NAME[model_name]
        color   = COLOR_MAP[model_name]
        handles.append(Line2D([0], [0], color=color, linewidth=2, linestyle='-',
                               marker='o', markersize=5, label=f'{display}(R101)'))
        handles.append(Line2D([0], [0], color=color, linewidth=2, linestyle='--',
                               marker='o', markersize=5, label=f'{display}(R50)'))
    handles.append(Line2D([0], [0], color=COLOR_MAP['ViT'], linewidth=2, linestyle='-',
                           marker='o', markersize=5, label='ViT(small_patch16_224)'))
    return handles


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
# 레이아웃: 2행(task) × 3열(dataset), model_type별로 별도 figure
# 범례: figure 상단 중앙, subplot 행렬 너비에 맞춰 정렬

os.makedirs('/home/hail/pan/GCN/PIGNet/csvs/cls_graph', exist_ok=True)

for model_type in MODEL_TYPES:
    fig, axes = plt.subplots(len(TASKS), len(DATASETS), figsize=(24, 13))

    for row_i, task in enumerate(TASKS):
        for col_i, dataset in enumerate(DATASETS):
            ax = axes[row_i, col_i]
            draw_subplot(ax, dataset, task, model_type, is_left_col=(col_i == 0))

            if row_i == 0:
                ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=FS_TITLE, fontweight='bold', fontfamily='Times New Roman', pad=18)

    plt.tight_layout()
    fig.subplots_adjust(top=0.80, left=0.1, hspace=0.5)

    # 범례를 subplot 행렬 가로 중앙에 배치
    center_x = (axes[0, 0].get_position().x0 + axes[0, -1].get_position().x1) / 2
    fig.legend(handles=make_legend_handles(),
               loc='upper center',
               bbox_to_anchor=(center_x, 1.0),
               bbox_transform=fig.transFigure,
               ncol=3,
               fontsize=FS_LEGEND,
               handlelength=3.5, handletextpad=0.8,
               borderpad=0.6, labelspacing=0.4, columnspacing=3.5,
               frameon=False,
               prop={'family': 'Times New Roman', 'size': FS_LEGEND})

    save_path = f'/home/hail/pan/GCN/PIGNet/csvs/cls_graph/{model_type}_combined.pdf'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Saved → {save_path}")
