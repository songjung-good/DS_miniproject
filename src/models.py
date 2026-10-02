"""고정된 Train CV로 선택하고 Hold-out 및 Batch 2에 한 번 평가한다."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_SETS, build_features


def metrics(y, predicted):
    return {"MAPE_pct": float(mean_absolute_percentage_error(y, predicted) * 100),
            "MAE_cycles": float(mean_absolute_error(y, predicted)),
            "RMSE_cycles": float(np.sqrt(mean_squared_error(y, predicted)))}


def reporting_table(scores):
    """점수(%)와 뒤 단계−앞 단계 Gap(%p)을 필수 6행 표로 묶는다."""
    rows = scores.set_index("split").loc[["Train_CV", "Valid", "Test_Batch2"]].reset_index().copy()
    rows["value"] = rows.MAPE_pct
    rows["unit"] = "%"
    train, valid, test = rows.MAPE_pct
    gaps = pd.DataFrame({
        "split": ["Valid_minus_TrainCV", "Test_minus_Valid", "Test_minus_9.1"],
        "value": [valid - train, test - valid, test - 9.1],
        "unit": "%p",
        "candidate_id": rows.candidate_id.iloc[0],
    })
    return pd.concat([rows, gaps], ignore_index=True)


def candidates():
    specs = [{"id": "Dummy_median", "feature_set": "A_summary", "model": "Dummy", "params": {}}]
    for group in FEATURE_SETS:
        for alpha in (0.1, 1.0, 10.0):
            specs.append({"id": f"Ridge_{group}_alpha{alpha}", "feature_set": group,
                          "model": "Ridge", "params": {"alpha": alpha}})
        for depth in (2, 3):
            for leaf in (3, 5):
                specs.append({"id": f"RF_{group}_depth{depth}_leaf{leaf}", "feature_set": group,
                              "model": "RandomForest", "params": {"max_depth": depth, "min_samples_leaf": leaf}})
    return specs


def pipeline(spec):
    if spec["model"] == "Dummy":
        return Pipeline([("model", DummyRegressor(strategy="median"))])
    if spec["model"] == "Ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(**spec["params"]))])
    return Pipeline([("model", RandomForestRegressor(n_estimators=200, random_state=42,
                                                      n_jobs=1, **spec["params"]))])


def run_experiment(root):
    root = Path(root)
    out = root / "results/modeling"
    if (out / "test_evaluation.json").exists():
        raise FileExistsError("Batch 2 평가가 이미 완료됐습니다. 저장 결과를 읽으세요.")
    out.mkdir(parents=True, exist_ok=True)
    specs = candidates()
    (out / "candidate_plan.json").write_text(json.dumps({"seed": 42, "cv_folds": 3,
        "selection": "minimum Train CV mean MAPE, then std, then candidate ID",
        "target": "cycle_life (원 단위)", "feature_sets": FEATURE_SETS, "candidates": specs}, ensure_ascii=False, indent=2))
    data, audit = build_features(root / "data/2017-05-12_batchdata_updated_struct_errorcorrect.mat", "Batch1")
    split = pd.read_csv(root / "results/train_val_split_cells.csv")
    if len(data) != 46 or len(split) != 46 or not split.cell_id.is_unique:
        raise ValueError("Batch 1 셀 명세 오류")
    if set(data.cell_id) != set(split.cell_id):
        raise ValueError("분할 명세와 입력 ID 불일치")
    data = data.merge(split[["cell_id", "split", "cycle_life"]], on="cell_id", validate="one_to_one", suffixes=("", "_manifest"))
    if not np.array_equal(data.cycle_life, data.cycle_life_manifest):
        raise ValueError("매니페스트 라벨 불일치")
    data = data.drop(columns="cycle_life_manifest")
    data.to_csv(out / "batch1_features.csv", index=False)
    audit.to_csv(out / "batch1_input_audit.csv", index=False)
    train = data[data.split == "train"].sort_values("cell_id").reset_index(drop=True)
    valid = data[data.split == "val"].sort_values("cell_id").reset_index(drop=True)
    assert len(train) == 36 and len(valid) == 10 and not set(train.cell_id) & set(valid.cell_id)
    folds = list(KFold(n_splits=3, shuffle=True, random_state=42).split(train))
    fold_assignment = np.zeros(len(train), dtype=int)
    for number, (_, held) in enumerate(folds, 1):
        fold_assignment[held] = number
    pd.DataFrame({"cell_id": train.cell_id, "cv_fold": fold_assignment}).to_csv(out / "cv_fold_cells.csv", index=False)
    scores, predictions = [], {}
    for spec in specs:
        columns = FEATURE_SETS[spec["feature_set"]]
        x, y = train[columns], train.cycle_life
        oof = np.full(len(train), np.nan)
        fold_scores = []
        estimator = pipeline(spec)
        for fitting, held in folds:
            fitted = clone(estimator).fit(x.iloc[fitting], y.iloc[fitting])
            oof[held] = fitted.predict(x.iloc[held])
            fold_scores.append(metrics(y.iloc[held], oof[held])["MAPE_pct"])
        assert np.isfinite(oof).all()
        predictions[spec["id"]] = oof
        scores.append({"candidate_id": spec["id"], "model": spec["model"], "feature_set": spec["feature_set"],
                       "parameters": json.dumps(spec["params"], sort_keys=True),
                       "CV_MAPE_mean_pct": float(np.mean(fold_scores)),
                       "CV_MAPE_std_pct": float(np.std(fold_scores, ddof=1)),
                       **{f"fold{i+1}_MAPE_pct": s for i, s in enumerate(fold_scores)}})
    comparison = pd.DataFrame(scores).sort_values(["CV_MAPE_mean_pct", "CV_MAPE_std_pct", "candidate_id"])
    comparison.to_csv(out / "cv_candidate_results.csv", index=False)
    selected_id = comparison.iloc[0].candidate_id
    selected = next(s for s in specs if s["id"] == selected_id)
    columns = FEATURE_SETS[selected["feature_set"]]
    frozen = {**selected, "features": columns, "train_cells": train.cell_id.tolist(),
              "selected_at_utc": datetime.now(timezone.utc).isoformat(), "selection_source": "Train 3-fold CV only"}
    (out / "selected_model.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    fitted = pipeline(selected).fit(train[columns], train.cycle_life)
    joblib.dump(fitted, out / "selected_model.joblib")
    # Validation/Test 확인 전에 선택과 Train 적합을 완료한다.
    valid_pred = fitted.predict(valid[columns])
    test, test_audit = build_features(root / "data/2018-02-20_batchdata_updated_struct_errorcorrect.mat", "Batch2")
    assert len(test) == 39
    test.to_csv(out / "batch2_features.csv", index=False)
    test_audit.to_csv(out / "batch2_input_audit.csv", index=False)
    test_pred = fitted.predict(test[columns])
    cv_pred = predictions[selected_id]
    result = pd.DataFrame([
        {"split": "Train_CV", "n_cells": 36, **metrics(train.cycle_life, cv_pred)},
        {"split": "Valid", "n_cells": 10, **metrics(valid.cycle_life, valid_pred)},
        {"split": "Test_Batch2", "n_cells": 39, **metrics(test.cycle_life, test_pred)},
    ])
    result["candidate_id"] = selected_id
    report = reporting_table(result)
    report.to_csv(root / "results/model_performance.csv", index=False)
    pred_frames = []
    for role, part, pred in [("Train_CV", train, cv_pred), ("Valid", valid, valid_pred), ("Test_Batch2", test, test_pred)]:
        frame = part[["batch", "cell_id", "policy", "cycle_life"]].copy()
        frame["split"] = role
        frame["prediction"] = pred
        frame["error_cycles"] = pred - frame.cycle_life
        frame["APE_pct"] = abs(frame.error_cycles) / frame.cycle_life * 100
        pred_frames.append(frame)
    pd.concat(pred_frames, ignore_index=True).to_csv(out / "cell_predictions.csv", index=False)
    test_score = result.loc[result.split == "Test_Batch2", "MAPE_pct"].iloc[0]
    report.loc[report.unit == "%p", ["split", "value"]].rename(
        columns={"split": "gap", "value": "gap_percentage_points"}
    ).to_csv(out / "performance_gaps.csv", index=False)
    (out / "test_evaluation.json").write_text(json.dumps({"evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": selected_id, "test_cells": 39, "n_test_evaluations": 1,
        "test_mape_pct": test_score, "rule": "No candidate selection or retuning after test"}, indent=2))
    return comparison, result


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    if (project_root / "results/modeling/test_evaluation.json").exists():
        print("이미 완료된 Batch 2 평가는 반복하지 않고 저장된 결과를 표시합니다.")
        performance = pd.read_csv(project_root / "results/model_performance.csv")
    else:
        _, performance = run_experiment(project_root)
    print(performance.to_string(index=False))
