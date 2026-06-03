# PIGNet Project

## 2026-06-01 작업 내용

### JE (Joint Entropy) Seg 실험 진행 상황

`JE_seg.sh` 스크립트로 3개 모델 × 2 backbone × 2 dataset × 2 model_type = **총 24회** JE 계산 실행.

**모델 목록:** `PIGNet_GSPonly` | `ASPP` | `Mask2Former`

**캐시 저장 경로:** `/home/hail/pan/HDD/IB_dataset/{dataset}/{backbone}/{model_type}/{model}/zoom/1/`
- `analysis_cache_same_diff_joint.pkl`
- `kde_cache_contour.pkl`

**완료 현황 (2026-06-01 기준):**
- ✅ PIGNet_GSPonly — 8개 조합 완료
- 🔄 ASPP — 실행 중 (당일 저녁 실행)
- ❌ Mask2Former — 미실행

**다음 할 일:**
- ASPP 완료 후 `JE_seg.sh`의 `MODEL="Mask2Former"`로 변경하여 실행
- 또는 MODEL을 루프에 추가해서 한 번에 3개 모델 모두 처리하도록 스크립트 개선 가능

### JE_seg.sh 주의사항

현재 `MODEL`이 하드코딩되어 있어 모델 1개씩 수동으로 변경 후 실행해야 함.
```bash
MODEL="ASPP"  # 여기를 바꿔서 재실행
```
