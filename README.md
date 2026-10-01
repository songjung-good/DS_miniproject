# ESS 배터리 수명 예측

배터리의 초기 충·방전 데이터를 이용해 셀의 수명을 예측하고, 다른 실험 배치에서도 예측이 유효한지 평가합니다. EDA에서 발견한 특성을 피처와 모델 설계에 연결하고, 배터리 선별·점검·교체 계획에 대한 활용 가능성을 해석합니다.

## 프로젝트 개요

- 데이터셋: MIT–Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습·검증 데이터: Batch 1 (2017-05-12)
- 최종 평가 데이터: Batch 2 (2018-02-20)
- 추가 데이터: Batch 3 (2018-04-12), EDA에 포함하며 모델 추가 평가는 미정
- 태스크: 회귀 또는 분류 중 선택 예정
- 검증 방식: CV 없이 셀 단위 Hold-out. 과제의 CV 평균 보고 항목은 미실시로 명시합니다.

현재는 프로젝트 설계와 예제 노트북 검토 단계입니다. 아래 분석 결과와 모델 성능은 실제 실행 후 작성합니다.

## 파일 구조

```text
├── data/                            # 데이터 파일 (대용량 원본 .mat은 Git 제외)
│   └── README.md
├── notebooks/                       # 단계별 Jupyter 노트북
├── src/                             # 전처리, 피처 추출, 모델 학습 모듈
├── results/                         # 모델 평가 지표 및 산출물
├── docs/                            # 프로젝트 개요, 과제 가이드, 작업 기록
├── requirements.txt                 # 패키지 의존성
└── README.md
```

분석은 `notebooks/` 내 `.ipynb` 형식으로 작성합니다. 각 노트북에 분석 목적, 코드, 그래프, 관측 결과와 해석을 함께 담습니다.

## 환경 설정

필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
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

노트북을 열고 데이터 경로를 설정한 뒤 셀을 위에서 아래로 실행합니다. 현재 참고 노트북에 저장된 출력과 해석은 이번 프로젝트에서 새로 검증한 결과가 아닙니다.

## EDA

Batch 1·2·3을 비교하고, 각 질문의 관측 결과와 모델 설계에 반영할 내용을 작성합니다.

- **Cycle Life 분포:** 분포 형태, 장·단수명 비율, 수명이 유독 짧은 셀 분석 — 결과 작성 예정
- **열화 곡선:** 셀별 용량 감소, 열화 속도, knee point 분석 — 결과 작성 예정
- **ΔQ(V) 곡선:** `Q100(V) - Q10(V)`와 장·단수명 셀의 차이 분석 — 결과 작성 예정
- **충전 조건과 수명:** C-rate·충전 프로토콜별 수명 비교 — 결과 작성 예정
- **초기 신호와 수명:** 초기 피처의 상관관계와 다중공선성 분석 — 결과 작성 예정

## Modeling

### 피처 엔지니어링 전략

EDA 결과를 근거로 피처를 선정할 예정입니다. 회귀는 초기 100사이클, 분류는 초기 5사이클을 사용하며, 분류 입력에 `Q100 - Q10`을 포함하지 않습니다.

### 모델 선택 및 근거

- 후보 모델: 미정
- 최종 모델: 미정
- 선택 이유: EDA와 동일한 Hold-out 분할의 후보 모델 비교 결과를 바탕으로 작성 예정

동일 셀의 사이클은 학습·검증 양쪽에 나누지 않습니다. 전처리는 학습 데이터에서만 학습하고, Batch 2를 이용해 모델을 튜닝하지 않습니다.

## 성능 결과

모델 평가 후 과제 양식에 맞춰 작성합니다.

- Train (Batch 1 CV): 미실시
- Valid (Batch 1 Hold-out): 평가 예정
- Test (Batch 2): 평가 예정
- Gap (Train-Valid): CV 미실시로 산출하지 않음
- Gap (Valid-Test), Gap (Target-Test): 평가 후 산출

회귀 선택 시 MAPE(%), 분류 선택 시 F1-Score와 Accuracy를 보고합니다. 목표 수치, Gap 계산식 및 F1 정의는 태스크 확정 후 명시합니다.

## 오류 분석

- 예측 오차가 큰 셀의 특성과 충전 조건: 평가 후 작성
- 원인 가설 및 개선 방향: 평가 후 작성

## ESS 도메인 해석

- 활용 가능성: 배터리 셀 선별, 점검 우선순위 및 교체 계획 지원
- 한계: 실험용 셀의 수명 예측을 ESS 전체 수명으로 직접 해석할 수 없습니다.
- 추가 검증: 실제 운전 조건, 온도, 화학 조성 및 셀 간 불균형을 고려한 검증이 필요합니다.

## 참고문헌

- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. _Nature Energy_, 4, 383–391.
- 데이터 다운로드: Kaggle `itshpark/data-driven-prediction-of-battery-cycle`
