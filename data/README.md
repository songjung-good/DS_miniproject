# Dataset

- Dataset source: MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- Download: Kaggle dataset `itshpark/data-driven-prediction-of-battery-cycle`

Files placed here:
- `2017-05-12_batchdata_updated_struct_errorcorrect.mat` (Batch 1: Train & Hold-out)
- `2018-02-20_batchdata_updated_struct_errorcorrect.mat` (과제 Batch 2 평가용; 논문의 2017-06-30 primary test와 구분)
- `2018-04-12_batchdata_updated_struct_errorcorrect.mat` (Batch 3: Secondary Test / EDA)
- `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat` (Extra: VarCharge study)

2026-10-02 출처 확인: Kaggle v1(2024-04-18)의 파일 목록과 로컬 4개 파일명·바이트 수가 일치한다. 체크섬에 의한 동일성은 검증하지 않았다. 배포 설명은 2018-02-20·2018-04-03 파일을 Figure 4 저율 진단 자료로 분류한다. 현재 Batch 1 라벨은 기록 수+1과 일치하고 기록에서 EOL이 관측되지 않아, 후속 측정 연결·라벨 처리 이력을 확인해야 한다. 현재 파일에 공식 2017-06-30의 연결 인덱스를 그대로 적용하지 않는다. 근거: `results/modeling/kaggle_view.json`, `kaggle_source_metadata.json`, `review_file_source.json`.
