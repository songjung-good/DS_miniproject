# ESS 배터리 수명 예측

배터리의 초기 충·방전 데이터를 이용해 셀의 수명을 예측하고, 다른 실험 배치에서도 예측이 유효한지 평가합니다. EDA에서 발견한 특성을 피처와 모델 설계에 연결하고, 배터리 선별·점검·교체 계획에 대한 활용 가능성을 해석합니다.

## 프로젝트 개요

- 데이터셋: MIT–Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 태스크: 초기 100사이클 기반 Cycle Life 회귀
- 검증 방식: Batch 1 Train 36셀의 3-fold CV, Hold-out 10셀, Batch 2 Test 39셀

### 원본 데이터 구성 (`data/`)

| 파일명 | 크기 | 식별명 | 셀 수 | 역할 및 적용 범위 |
| :--- | :--- | :--- | :---: | :--- |
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | ~2.8 GB | **Batch 1** | 46 | **Train & 내부 Hold-out 검증용** (필수) |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | ~1.9 GB | **Batch 2** | 47 | **최종 일반화 Test용** (필수) |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | ~3.0 GB | **Batch 3** | 46 | **추가 비교 분석용** (Batch 1·2 완료 후 진행) |
| `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat` | ~85 MB | **Extra** | - | **본 과제 제외** (충전 최적화 연구 데이터) |

*진행 원칙: Batch 1과 Batch 2를 기준으로 프로젝트 파이프라인(EDA → 피처 → 모델)을 완결한 뒤 Batch 3을 추가 평가에 활용합니다.*

**출처·타깃 해석 유의:** 과제의 Batch 2(2018-02-20)는 논문의 primary test(2017-06-30)와 다릅니다. Kaggle 설명은 현재 파일을 Figure 4 저율 진단 자료로 분류합니다. 또한 Batch 1의 제공 라벨은 모두 상세 기록 수+1과 일치하며 기록 내 EOL 도달은 없습니다. 현재 성능은 제공 라벨에 대한 기준 결과로 보존하고, 동일 조건의 실제 EOL 예측·논문 재현 성능으로 확정하지 않습니다. [출처·라벨 검토](docs/reports/DAY2_MODEL_RESULT.md).

## 파일 구조

```text
├── data/                            # 데이터 파일 (대용량 원본 .mat은 Git 제외)
│   └── README.md
├── notebooks/                       # 단계별 Jupyter 노트북
├── src/                             # 전처리, 피처 추출, 모델 학습 모듈
├── results/                         # 모델 평가 지표 및 산출물
├── docs/                            # 프로젝트 개요, 과제 가이드, 작업 기록
│   ├── reports/                     # 산출물 (설계 보고서, 제출용 HTML/PDF)
│   └── assets/                      # 참고 이미지 및 스크린샷
├── requirements.txt                 # 패키지 의존성
└── README.md
```

분석은 `notebooks/` 내 `.ipynb` 형식으로 작성합니다. 각 노트북에 분석 목적, 코드, 그래프, 관측 결과와 해석을 함께 담습니다.

## 환경 설정

`uv`를 사용하여 가상환경을 생성하고 의존성을 설치합니다.

```bash
# 1. 가상환경 생성 (Python 3.11 권장)
uv venv --python 3.11

# 2. 가상환경 활성화
source .venv/bin/activate

# 3. 의존성 설치
uv pip install -r requirements.txt

# 4. Jupyter 커널 등록
python -m ipykernel install --user --name ds_miniproject --display-name "Python (ds_miniproject)"
```

### 데이터 다운로드

대용량 원본 데이터는 `data/`에 보관하며 Git에 포함하지 않습니다. 필요할 때 아래 코드로 Kaggle 데이터셋을 다운로드할 수 있습니다.

```python
import kagglehub

path = kagglehub.dataset_download("itshpark/data-driven-prediction-of-battery-cycle")
print("Path to dataset files:", path)
```

분석 코드의 데이터 경로는 `data/`를 기본으로 사용합니다.

이 코드는 최신 버전을 받으므로 동일한 분석을 재현하려면 사용한 데이터셋 버전을 기록해야 합니다.

### 노트북 실행

```bash
python -m jupyter notebook
```

노트북을 열고 데이터 경로를 설정한 뒤 셀을 위에서 아래로 실행합니다. `01_EDA.ipynb`는 통합 EDA, `02_Modeling.ipynb`는 1차 모델 실행·평가 결과입니다. 이미 완료된 Batch 2 평가는 저장 결과를 읽습니다.

## EDA

Batch 1·2·3을 비교하고, 각 질문의 관측 결과와 모델 설계에 반영할 내용을 작성합니다.

- Q1·Q2: 배치별 수명 분포와 Batch 1 초기 총 방전용량 변화를 확인했습니다.
- Q3: ΔQ(V) 로그 분산은 수명과 연관되지만 정책 영향과 후보 간 중복을 확인했습니다.
- Q4: 첫 C-rate 단독으로 배치별 수명을 설명하기 어렵고 두 번째 C-rate·전환 SOC도 검토했습니다.
- Q5: 초기 요약 지표·곡선 통계와 수명 관계를 일부 검증했습니다. 인과관계는 확정하지 않았습니다.

## Modeling — 1차 검토 결과

기존 36:10 셀 분할을 유지하고 Train 내부 CV의 동일 fold로 Dummy·Ridge·얕은 RandomForest 22개 후보를 비교했습니다. CV 최저 MAPE로 고른 모델은 **StandardScaler + Ridge(alpha=0.1)**입니다. 입력은 용량 변화·최고온도·ΔQ(V) 로그 분산·첫 C-rate·두 번째 C-rate·전환 SOC의 6개입니다. 전처리는 fold별 학습 부분에서만 적합했습니다.

| 평가 대상 | 셀 수 | MAPE |
| --- | ---: | ---: |
| Train 3-fold CV | 36 | 9.88% |
| Validation | 10 | 8.93% |
| Batch 2 Test | 39 | 58.73% |

Valid−Train(CV 평균)은 −0.95%p, Test−Valid는 +49.80%p, Test−목표 9.1%는 +49.63%p입니다. 최종 Test 목표를 달성하지 못했습니다.

Batch 2 일반 표기 셀의 수명을 크게 과대예측했습니다. 일반 표기 30셀의 MAPE는 72.34%, newstructure 9셀은 13.38%입니다. 배치·정책·입력 범위 차이를 고려해야 하며, Test 결과를 보고 모델을 다시 튜닝하지 않았습니다. EDA에서 Batch 1 전체 라벨과 Batch 2 라벨 분포를 관측한 한계도 보고합니다.

[실행 결과 노트북](notebooks/02_Modeling.ipynb), [1차 결과 보고서](docs/reports/DAY2_MODEL_RESULT.md), [성능 CSV](results/model_performance.csv)를 참고하세요.

```bash
# 최초 실행은 피처 계산·CV 선택·고정 모델 평가, 이후 실행은 저장 성능 표시
python -m src.models

# 데이터 대응·전처리·점수 계산·재평가 방지 확인
python -m unittest discover -s tests -v
```

## ESS 도메인 해석

- 활용 가능성: 배터리 셀 선별, 점검 우선순위 및 교체 계획 지원
- 한계: 실험용 셀의 수명 예측을 ESS 전체 수명으로 직접 해석할 수 없습니다.
- 추가 검증: 실제 운전 조건, 온도, 화학 조성 및 셀 간 불균형을 고려한 검증이 필요합니다.

## 참고문헌

- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. _Nature Energy_, 4, 383–391.
- 데이터 다운로드: Kaggle `itshpark/data-driven-prediction-of-battery-cycle`
