import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'regular'

# ── 데이터 범위 설정 ──────────────────────────────────────────────────────────
# DATASETS : 열(column) 구성 — 파일 이름 및 subplot 제목에 직접 사용됨
# TASKS    : 행(row) 구성 — 각 행이 하나의 task에 대응
# MODEL_TYPES : 별도 figure로 출력; 파일명 및 Excel sheet 이름에 사용됨
DATASETS    = ['pascal', 'cityscape']
TASKS       = ['zoom', 'repeat', 'overlap']
MODEL_TYPES = ['scratch', 'pretrain']

# ── 시각화 스타일 ─────────────────────────────────────────────────────────────
# COLOR_MAP    : 모델별 선/밴드 색상
# LEGEND_ORDER : 범례 표시 순서 (위→아래, 왼→오른쪽)
# ZORDER_MAP   : 겹칠 때 위에 그려질 모델 우선순위 (값이 클수록 앞)
# DISPLAY_NAME : 범례에 실제로 표시되는 이름
COLOR_MAP     = {'PIGNet_GSPonly': '#D81B60', 'ASPP': '#1E88E5', 'Mask2Former': '#FFC107'}
LEGEND_ORDER  = ['PIGNet_GSPonly', 'ASPP', 'Mask2Former']
ZORDER_MAP    = {'PIGNet_GSPonly': 10, 'Mask2Former': 5, 'ASPP': 3}
DISPLAY_NAME  = {'PIGNet_GSPonly': 'PIGNet_GSPOnly', 'ASPP': 'ASPP', 'Mask2Former': 'Mask2Former'}

# x축 라벨 텍스트 (task별)
TASK_XLABEL = {'zoom': 'Zoom Ratio', 'repeat': 'Repeat count', 'overlap': 'Overlap Ratio'}

# ── 폰트 크기 ────────────────────────────────────────────────────────────────
# 아래 값을 줄이면 해당 요소 텍스트가 작아짐
FS_TITLE  = 35   # subplot 상단 제목 (Pascal / Cityscape)
FS_LABEL  = 25   # x·y축 라벨
FS_TICK   = 20   # x·y축 눈금 숫자
FS_LEGEND = 30   # 범례 텍스트


# ── 헬퍼 함수 ────────────────────────────────────────────────────────────────

def normalize_model_name(name):
    """Excel 시트의 원본 모델명 → 내부 키(COLOR_MAP 등)로 변환"""
    m = str(name).lower()
    if 'aspp'        in m: return 'ASPP'
    if 'gsp'         in m: return 'PIGNet_GSPonly'
    if 'mask2former' in m: return 'Mask2Former'
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
    """첫 번째 열에서 모델명 목록을 순서대로 추출 (ratio/task명 제외)."""
    skip = {'ratio', 'zoom', 'repeat', 'overlap'}
    seen, out = set(), []
    for m in df.iloc[:, 0].dropna().astype(str):
        clean = m.strip()
        if clean.lower() not in skip and clean not in seen:
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
                d[rv] = (mv, sv)
        data[str(model)] = d
    return data


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
    all_ratios = set()
    for d in (data_101, data_50):
        for v in d.values():
            all_ratios.update(v.keys())
    global_ratios = sorted(all_ratios)
    x_idx = np.arange(len(global_ratios))
    r2i   = {r: i for i, r in enumerate(global_ratios)}

    # LEGEND_ORDER 기준으로 모델 정렬 → 중요 모델이 마지막에 그려져 앞에 표시됨
    all_models = sorted(
        set(models_101) | set(models_50),
        key=lambda x: LEGEND_ORDER.index(normalize_model_name(x))
                      if normalize_model_name(x) in LEGEND_ORDER else 99
    )

    # R101=실선, R50=점선으로 구분
    for data_dict, models, backbone in [(data_101, models_101, '101'),
                                         (data_50,  models_50,  '50')]:
        linestyle = '-' if backbone == '101' else '--'
        for model_orig in all_models:
            if model_orig not in models:
                continue
            model_name = normalize_model_name(model_orig)
            ratio_data = data_dict.get(model_orig, {})

            px, py, ps = [], [], []
            for r in global_ratios:
                if r in ratio_data:
                    px.append(r2i[r])
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
    ax.set_xticks(x_idx)
    if task == 'zoom':
        tick_labels = [f"{int(r)}%" for r in global_ratios]
    else:
        tick_labels = [str(int(r) if r == int(r) else r) for r in global_ratios]
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', rotation_mode='anchor',
                       fontsize=FS_TICK, fontfamily='Times New Roman')
    ax.set_xlabel(TASK_XLABEL[task], fontsize=FS_LABEL, fontfamily='Times New Roman')

    # y축: 왼쪽 열만 라벨 표시, 항상 0에서 시작
    if is_left_col:
        ax.set_ylabel('mIoU (%)', fontsize=FS_LABEL, fontfamily='Times New Roman')
    else:
        ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=FS_TICK, labelleft=True)
    ylim_top = 40 if model_type == 'scratch' else 75
    ax.set_ylim(0, ylim_top)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))  # n 단위마다 눈금

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)


def make_legend_handles():
    """
    범례 핸들 생성. ncol=3이면 matplotlib은 열 우선(column-first)으로 채움:
        PIGNet(R101)  ASPP(R101)  Mask2Former(R101)
        PIGNet(R50)   ASPP(R50)   Mask2Former(R50)
    → 핸들 순서: PIGNet_R101, PIGNet_R50, ASPP_R101, ASPP_R50, ...
    """
    LINESTYLE = {'R101': '-', 'R50': '--'}
    handles = []
    for model_name in LEGEND_ORDER:
        for suffix in ['R101', 'R50']:
            display = DISPLAY_NAME.get(model_name, model_name)
            color   = COLOR_MAP[model_name]
            handles.append(Line2D([0], [0],
                                   color=color, linewidth=2,
                                   linestyle=LINESTYLE[suffix],
                                   marker='o', markersize=5,
                                   label=f'{display}({suffix})'))
    return handles


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
# 레이아웃: 3행(task) × 2열(dataset), model_type별로 별도 figure
# 범례: figure 상단 중앙, subplot 행렬 너비에 맞춰 정렬

for model_type in MODEL_TYPES:
    fig, axes = plt.subplots(len(TASKS), len(DATASETS), figsize=(16, 16))

    for row_i, task in enumerate(TASKS):
        for col_i, dataset in enumerate(DATASETS):
            ax = axes[row_i, col_i]
            draw_subplot(ax, dataset, task, model_type, is_left_col=(col_i == 0))

            if row_i == 0:
                ax.set_title(dataset.capitalize(), fontsize=FS_TITLE, fontweight='bold',
                             fontfamily='Times New Roman', pad=10)

    plt.tight_layout()
    fig.subplots_adjust(top=0.86, left=0.1)

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

    save_path = f'/home/hail/pan/GCN/PIGNet/csvs/seg_graph/{model_type}_combined.pdf'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Saved → {save_path}")
