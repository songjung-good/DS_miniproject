# DS_miniproject

## 데이터 다운로드

대용량 원본 데이터는 `archive/`에 보관하며 Git에 포함하지 않습니다.
필요할 때 아래 코드로 Kaggle 데이터셋을 다운로드할 수 있습니다.

```bash
python -m pip install kagglehub
```

```python
import kagglehub

path = kagglehub.dataset_download("itshpark/data-driven-prediction-of-battery-cycle")
print("Path to dataset files:", path)
```

분석 노트북 `docs/30-ESSHealth-scratch.ipynb`의 `DATA_DIR`을 출력된 경로로 설정하세요.
다운로드 경로가 `archive/`라고 가정하지 않고, 반환된 `path`를 사용합니다.
이 코드는 최신 버전을 받으므로 동일한 분석을 재현하려면 사용한 데이터셋 버전을 기록하세요.
