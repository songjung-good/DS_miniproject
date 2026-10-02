"""초기 100사이클의 모델 입력을 원본에서 계산한다."""
import re

import h5py
import numpy as np
import pandas as pd

from src.preprocess import load_batch_summary

POLICY = re.compile(r"^(\d+(?:\.\d+)?)C\((\d+(?:\.\d+)?)%\)-(\d+(?:\.\d+)?)C(?:-newstructure)?$")
FEATURE_SETS = {
    "A_summary": ["q_diff_100_2", "t_max_100"],
    "B_voltage": ["q_diff_100_2", "t_max_100", "delta_q_log_var"],
    "C_policy": ["q_diff_100_2", "t_max_100", "delta_q_log_var",
                 "charge_c1_rate", "charge_c2_rate", "charge_switch_soc"],
}


def build_features(path, batch_name):
    """라벨 보유 셀을 계산한다. 비정상 입력은 보간하지 않고 오류로 알린다."""
    summary = load_batch_summary(path, batch_name)
    records, audits = [], []
    with h5py.File(path, "r") as f:
        batch = f["batch"]
        for _, row in summary.iterrows():
            valid_label = bool(np.isfinite(row.cycle_life) and row.cycle_life > 0)
            if not valid_label:
                audits.append({"batch": batch_name, "cell_id": row.cell_id,
                               "status": "excluded_missing_label"})
                continue
            positions = {}
            for cycle in (2, 10, 100):
                hits = np.flatnonzero(row.cycles == cycle)
                if len(hits) != 1:
                    raise ValueError(f"{row.cell_id}: Cycle {cycle} 누락/중복")
                positions[cycle] = int(hits[0])
            early = (row.cycles >= 2) & (row.cycles <= 100)
            if not np.array_equal(row.cycles[early], np.arange(2, 101)):
                raise ValueError(f"{row.cell_id}: Cycle 2~100 불연속")
            detail = f[batch["cycles"][int(row.cell_idx), 0]]
            voltage = f[batch["Vdlin"][int(row.cell_idx), 0]][()].ravel()
            if detail["Qdlin"].shape[0] != len(row.cycles):
                raise ValueError(f"{row.cell_id}: 상세/summary 길이 불일치")
            if not np.isfinite(voltage).all() or not (np.all(np.diff(voltage) < 0) or np.all(np.diff(voltage) > 0)):
                raise ValueError(f"{row.cell_id}: 전압 축 오류")
            qs = []
            for cycle in (10, 100):
                j = positions[cycle]
                q = f[detail["Qdlin"][j, 0]][()].ravel()
                raw = f[detail["Qd"][j, 0]][()].ravel()
                if q.shape != voltage.shape or not np.isfinite(q).all():
                    raise ValueError(f"{row.cell_id}: Qdlin 배열 오류")
                if not np.isclose(raw.max(), row.QDischarge[j], rtol=0, atol=1e-6):
                    raise ValueError(f"{row.cell_id}: 사이클 대응 오류")
                qs.append(q)
            variance = float(np.var(qs[1] - qs[0], ddof=0))
            if not np.isfinite(variance) or variance <= 0:
                raise ValueError(f"{row.cell_id}: 분산 0 또는 비유한")
            match = POLICY.fullmatch(row.policy)
            if not match:
                raise ValueError(f"{row.cell_id}: 미해석 정책 {row.policy}")
            c1, soc, c2 = map(float, match.groups())
            values = {
                "q_diff_100_2": row.QDischarge[positions[100]] - row.QDischarge[positions[2]],
                "t_max_100": float(np.max(row.Tmax[early])),
                "delta_q_log_var": np.log10(variance),
                "charge_c1_rate": c1, "charge_c2_rate": c2, "charge_switch_soc": soc,
            }
            if not np.isfinite(list(values.values())).all():
                raise ValueError(f"{row.cell_id}: 비유한 입력")
            records.append({"batch": batch_name, "cell_id": row.cell_id, "policy": row.policy,
                            "cycle_life": row.cycle_life, **values})
            audits.append({"batch": batch_name, "cell_id": row.cell_id, "status": "passed"})
    result = pd.DataFrame(records)
    if not result.cell_id.is_unique:
        raise ValueError("중복 셀 ID")
    return result, pd.DataFrame(audits)
