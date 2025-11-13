# GSP + SPP(가제)
-----------------------
## 개요(Overview)
-----------------------
PIGNet은 이미지를 픽셀 그래프로 변환해 그래프 합성곱(GCN)으로 처리하는 경량화 된 CV모델입니다.

<img width="624" height="286" alt="스크린샷 2025-08-11 오후 8 22 45" src="https://github.com/user-attachments/assets/4aa84afd-62de-4c96-950e-6eadd9fbd5d2" />

PIGNet은 Back-bone을 기존의 CN기반 모델을 이용하여 input 이미지의 latent를 생성하고, latent를 GSP와 SPP를 통해 각각의 노드에 global, local정보를 효율적으로 전달함으로서 주어진 Task를 완수 할 수 있습니다.
실험은 CV Task 중 Segmentation, Classfication을 진행하였고 그 결과로 기존의 모델과 동일하거나 다소 우세한 성능을 보이는 것을 확인하였으며 또한 확연한 파라미터 수의 감소를 수치화 함으로서 PIGNet의 우수성을 증명하였습니다.
