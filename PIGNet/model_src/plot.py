import matplotlib.pyplot as plt
def seg_zoom_plot():
    # 줌 비율 (X축 값)
    zoom_levels = [0.1, 0.5, 1.5, 2]

    # 각 모델에 대한 성능 데이터 (Y축 값)
    pignet101 = [3.63, 58.42, 64.45, 43.60]
    asap = [8.15, 64.53, 74.91, 72.01]
    mask2former = [3.1, 3.9, 7, 5.4]

    # 그래프 그리기
    plt.figure(figsize=(10, 6))

    plt.plot(zoom_levels, pignet101, label='Pignet101', marker='o')
    plt.plot(zoom_levels, asap, label='Asap', marker='o')
    plt.plot(zoom_levels, mask2former, label='Mask2former', marker='o')

    # 그래프 설정
    plt.title('Performance across Different Zoom Levels')
    plt.xlabel('Zoom Levels')
    plt.ylabel('mIoU')
    plt.grid(True)
    plt.legend()

    # 그래프 보여주기
    plt.show()


def seg_overlap_plot():
    # 줌 비율 (X축 값)
    zoom_levels_overlap = [0.1, 0.3, 0.6, 0.7]

    # 각 모델에 대한 overlap 성능 데이터 (Y축 값)
    pignet101_overlap = [19.91, 26.93, 27.82, 25.61]
    asap_overlap = [31.95, 28.94, 30.22, 30.43]
    mask2former_overlap = [3.69, 3.69, 3.86, 3.9]

    # 그래프 그리기
    plt.figure(figsize=(10, 6))

    plt.plot(zoom_levels_overlap, pignet101_overlap, label='Pignet101 Overlap', marker='o')
    plt.plot(zoom_levels_overlap, asap_overlap, label='Asap Overlap', marker='o')
    plt.plot(zoom_levels_overlap, mask2former_overlap, label='Mask2former Overlap', marker='o')

    # 그래프 설정
    plt.title('Overlap Performance across Different Zoom Levels')
    plt.xlabel('Zoom Levels')
    plt.ylabel('mIoU')
    plt.grid(True)
    plt.legend()

    # 그래프 보여주기
    plt.show()

def seg_repeat_plot():
    # 반복 횟수 (X축 값)
    repeat_levels = [0.25,0.11, 0.06 ,0.02] # 2 번,3번,4번,6번 counter 증가

    # 각 모델에 대한 repeat 성능 데이터 (Y축 값)
    pignet101_repeat = [16.86, 8.83, 3.65, 3.66]
    asap_repeat = [18.85, 9.34, 3.78, 3.67]
    mask2former_repeat = []  # 데이터가 없는 경우

    # 그래프 그리기
    plt.figure(figsize=(10, 6))

    plt.plot(repeat_levels, pignet101_repeat, label='Pignet101 Repeat', marker='o')
    plt.plot(repeat_levels, asap_repeat, label='Asap Repeat', marker='o')

    if mask2former_repeat:  # 데이터가 있을 경우만 그래프 그리기
        plt.plot(repeat_levels, mask2former_repeat, label='Mask2former Repeat', marker='o')

    # 그래프 설정
    plt.title('Repeat Performance across Different Levels')
    plt.xlabel('Repeat Levels')
    plt.ylabel('mIoU')
    plt.grid(True)
    plt.legend()

    # 그래프 보여주기
    plt.show()

