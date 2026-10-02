# 2일차 모델 개발·평가 — 1차 검토 결과

2026-10-02. 초기 100사이클 회귀 설계를 구현하고 모델 비교·Batch 2 평가까지 실행했다. 공동 검토를 위한 첫 결과이며, 실제 ESS 적용이나 목표 달성을 주장하지 않는다.

## 1. 어떤 순서로 진행했는가

1. 원본에서 셀당 한 행의 피처를 계산했다. Batch 1 46셀, Batch 2 라벨 보유 39셀은 계산 검증을 통과했고 Batch 2 라벨 결측 8셀은 평가에서 분리했다.
2. 기존 명세의 Train 36셀·Validation 10셀을 유지했다. Train 내부 3-fold shuffled CV(seed 42)의 같은 fold를 후보 22개에 적용했다. 매 fold는 24셀 적합·12셀 평가다.
3. 후보·파라미터·피처 구성을 먼저 기록했다. 최저 Train CV MAPE 평균을 기준으로 선택하고 Train 36셀에 적합한 모델을 저장했다.
4. 선택 후 Validation 10셀, 이어서 Batch 2 39셀에 평가했다. Test를 보고 모델·피처를 변경하지 않았다. 재실행 시 저장 결과를 읽어 재평가를 방지한다.

Ridge의 StandardScaler는 각 fold 학습 부분에서만 적합했다. 최종 스케일러는 Train 36셀에서 적합했다. 타깃 `cycle_life`와 실험 종료 기록량은 입력에서 제외했다. 원본 특이값의 임의 클리핑·대체는 도입하지 않았다.

## 2. 피처·모델 비교와 선택

| 구성 | 입력 | 최저 Train CV MAPE | 선택된 후보 |
| --- | --- | ---: | --- |
| 기준 모델 | Train 수명 중앙값 | 20.61% | DummyRegressor |
| A | 총 용량 변화·초기 최고온도 | 16.87% | Ridge alpha=10 |
| B | A + ΔQ(V) 로그 분산 | 12.07% | Ridge alpha=1 |
| C | B + 첫 C-rate·두 번째 C-rate·전환 SOC | 9.88% | Ridge alpha=0.1 |

후보는 Dummy 1개, Ridge 9개, 깊이 제한 RandomForest 12개다. 최종 고정 모델은 **StandardScaler + Ridge(alpha=0.1)**이며 피처는 6개다. 목표는 원 단위 `cycle_life`이고 로그 타깃 변환은 적용하지 않았다.

Q3 전압 곡선 정보와 Q4 정책 변수를 더했을 때 내부 점수가 개선됐다. 그러나 여러 후보에서 고른 최소 점수는 선택 편향이 있고, 작은 고정 분할의 개선이 새 배치의 개선을 보장하지 않는다. CV fold MAPE는 11.03%, 8.82%, 9.79%, 표본 표준편차는 1.11%p다.

![피처 구성별 CV 비교](../../results/modeling/cv_feature_comparison.png)

## 3. 고정 모델의 평가 결과

| 대상 | 셀 수 | MAPE | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Train CV | 36 | 9.88% | 78.63사이클 | 98.45사이클 |
| Validation | 10 | 8.93% | 84.67사이클 | 103.63사이클 |
| Batch 2 Test | 39 | **58.73%** | 274.04사이클 | 313.69사이클 |

Train CV는 학습셋 재예측 점수가 아니라 fold 밖 예측이다. 최종 모델은 Train 36셀에만 적합했다.

- Valid−Train(CV 평균): −0.95%p
- Test−Valid: +49.80%p
- Test−목표 9.1%: +49.63%p

Validation이 9.1%보다 낮아도 최종 Test에서 목표를 달성한 것은 아니다. 논문과 배치·분할 조건이 다르므로 직접 동등한 재현 실험으로 해석하지 않는다.

![실제 수명과 예측](../../results/modeling/actual_vs_prediction.png)

## 4. 어떤 실패가 확인됐는가

Batch 2의 일반 표기 30셀은 MAPE 72.34%, newstructure 표기 9셀은 13.38%다. 일반 표기 셀은 평균 약 320사이클 과대예측됐다. `Batch2_c15`는 실제 396사이클인데 약 1,133사이클로 예측되어 APE가 186.02%였다.

학습 수명 최저는 534사이클이고 Batch 2의 짧은 수명 영역은 학습에서 충분히 관측되지 않았다. Train 피처 범위를 벗어난 Test 셀은 용량 변화 12개, 최고온도 8개, 로그 분산 11개, 두 번째 C-rate 4개, 전환 SOC 2개다. 각 변수의 범위 안에 들어오는 셀도 결합 조건·정책은 다를 수 있다. 범위 차이나 정책별 오차만으로 실험상 원인을 확정하지 않는다.

어제의 배치·정책 분포 차이와 Q3·Q4의 정책 영향 우려가 실제 일반화 성능에서도 드러났다. 짧은 Test 셀을 제외하거나 같은 Test 결과로 모델을 다시 고르면 평가 의미가 바뀐다.

## 5. 함께 검토할 사항과 배운 점

- Q3·Q4 데이터를 계산하는 방식은 독립 EDA 결과와 일치했고, 최종 스케일러가 Train에서만 적합됐는지 확인했다.
- 강한 상관과 낮은 내부 CV 점수만으로 새 배치의 성능을 보장할 수 없음을 확인했다. 모델 복잡도보다 정책·실험 조건의 차이를 이해하는 일이 중요하다.
- 이후 개선 실험은 정책 그룹 검증, 짧은 수명 영역을 포함한 추가 학습 데이터, newstructure의 실험상 의미 확인 등을 검토한다. 현재 Test를 관측한 이후의 탐색이라는 사실을 보고한다.
- Batch 1 전체 라벨이 EDA에 사용됐고 정책이 Train/Validation에 겹친다. Batch 2 라벨 분포도 Q1에서 관측했으므로 완전 블라인드 성능이라고 표현하지 않는다.
- 이번 모델·선택 기록·Test 결과는 보존하고, 공동 검토에서 새 실험을 정하면 별도 버전으로 진행한다.

## 6. 검토·재현 자료

- [실행 결과 노트북](../../notebooks/02_Modeling.ipynb)
- [피처 구현](../../src/features.py), [모델 비교·평가 구현](../../src/models.py)
- [후보 CV 전체 표](../../results/modeling/cv_candidate_results.csv)
- [최종 성능 표](../../results/model_performance.csv), [셀별 예측·오차](../../results/modeling/cell_predictions.csv)
- [선택 기록](../../results/modeling/selected_model.json), [Test 평가 기록](../../results/modeling/test_evaluation.json)

프로젝트 루트에서 `python -m src.models`를 실행하면 평가 기록이 없을 때 전체 과정을 실행하고, 이미 완료됐다면 저장 성능만 표시한다. 원본 데이터 없이도 노트북의 저장된 결과를 검토할 수 있다. `python -m unittest discover -s tests -v`로 데이터 대응·스케일러·점수 계산·재평가 방지를 확인한다.
