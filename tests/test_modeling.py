"""1차 결과의 데이터 대응·누수 방지·성능 계산·재평가 방지를 확인한다."""
import json
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from src.models import reporting_table, run_experiment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/modeling"


class ModelingChecks(unittest.TestCase):
    def test_reporting_table_preserves_scores_and_gap_units(self):
        saved = pd.read_csv(ROOT / "results/model_performance.csv")
        report = reporting_table(saved)
        self.assertEqual(len(saved), 6)
        self.assertEqual(report.unit.tolist(), ["%", "%", "%", "%p", "%p", "%p"])
        np.testing.assert_allclose(saved.value, report.value, atol=1e-12)
        gaps = pd.read_csv(OUT / "performance_gaps.csv").set_index("gap")
        for _, row in report[report.unit == "%p"].iterrows():
            self.assertAlmostEqual(row.value, gaps.loc[row.split, "gap_percentage_points"], places=10)
        self.assertTrue(saved.loc[saved.unit == "%p", "MAPE_pct"].isna().all())

    def test_features_match_independent_eda(self):
        eda = pd.read_csv(ROOT / "results/q3_q4/q3_cell_features.csv").set_index("cell_id")
        policies = pd.read_csv(ROOT / "results/q3_q4/q4_cell_features.csv").set_index("cell_id")
        for batch in ("batch1", "batch2"):
            features = pd.read_csv(OUT / f"{batch}_features.csv").set_index("cell_id")
            np.testing.assert_allclose(features.delta_q_log_var, eda.loc[features.index, "delta_q_log_var"], rtol=1e-12)
            np.testing.assert_allclose(features.q_diff_100_2 * 1000, policies.loc[features.index, "delta_q_100_2_mAh"], atol=1e-10)
            for feature, source in [("charge_c1_rate", "c1"), ("charge_c2_rate", "c2"), ("charge_switch_soc", "switch_soc"), ("t_max_100", "tmax_2_100")]:
                np.testing.assert_allclose(features[feature], policies.loc[features.index, source], atol=1e-12)

    def test_scaler_fits_only_train(self):
        selected = json.loads((OUT / "selected_model.json").read_text())
        features = pd.read_csv(OUT / "batch1_features.csv")
        train = features[features.split == "train"]
        model = joblib.load(OUT / "selected_model.joblib")
        np.testing.assert_allclose(model["scale"].mean_, train[selected["features"]].mean(), atol=1e-12)
        self.assertEqual(model["scale"].n_samples_seen_, 36)
        self.assertEqual(set(selected["train_cells"]), set(train.cell_id))
        self.assertFalse({"cycle_life", "n_cycles"} & set(selected["features"]))

    def test_metrics_and_selection_match_saved_predictions(self):
        scores = pd.read_csv(ROOT / "results/model_performance.csv").set_index("split")
        predictions = pd.read_csv(OUT / "cell_predictions.csv")
        self.assertTrue(predictions.cell_id.is_unique)
        self.assertEqual(len(predictions), 85)
        for role, part in predictions.groupby("split"):
            score = 100 * mean_absolute_percentage_error(part.cycle_life, part.prediction)
            self.assertAlmostEqual(score, scores.loc[role, "MAPE_pct"], places=10)
        selected = json.loads((OUT / "selected_model.json").read_text())
        candidates = pd.read_csv(OUT / "cv_candidate_results.csv").sort_values(["CV_MAPE_mean_pct", "CV_MAPE_std_pct", "candidate_id"])
        self.assertEqual(selected["id"], candidates.iloc[0].candidate_id)

    def test_completed_test_is_not_repeated(self):
        ledger = OUT / "test_evaluation.json"
        before = ledger.read_bytes()
        with self.assertRaises(FileExistsError):
            run_experiment(ROOT)
        self.assertEqual(before, ledger.read_bytes())


if __name__ == "__main__":
    unittest.main()
