import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 경로 설정
project_root = "/Users/skala_yh/DS_miniproject"
results_dir = os.path.join(project_root, "results")
data_dir = os.path.join(project_root, "data")
os.makedirs(results_dir, exist_ok=True)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로더 (기존 preprocess/notebook 로더 활용)
import sys
sys.path.append(os.path.join(project_root, "src"))
from preprocess import load_batch_summary

b1_path = os.path.join(data_dir, "2017-05-12_batchdata_updated_struct_errorcorrect.mat")
b2_path = os.path.join(data_dir, "2018-02-20_batchdata_updated_struct_errorcorrect.mat")

df_b1 = load_batch_summary(b1_path, "Batch1")
df_b2 = load_batch_summary(b2_path, "Batch2")

# clean cells manifest 로드
clean_manifest = pd.read_csv(os.path.join(results_dir, "clean_cells_manifest.csv"))
clean_b1_ids = set(clean_manifest[clean_manifest["batch"] == "Batch1"]["cell_id"])
clean_b2_ids = set(clean_manifest[clean_manifest["batch"] == "Batch2"]["cell_id"])

df_b1_clean = df_b1[df_b1["cell_id"].isin(clean_b1_ids)].copy()
df_b2_clean = df_b2[df_b2["cell_id"].isin(clean_b2_ids)].copy()

print(f"Batch 1 clean cells: {len(df_b1_clean)}, Batch 2 clean cells: {len(df_b2_clean)}")

# Q2 분석: 방전용량 열화 곡선 (QDischarge vs Cycle)
# Knee-point 탐색: 각 셀별로 Qd 곡선의 곡률(Curvature) 또는 dQ/dCycle의 급변점 분석
# 간단하고 강건한 Knee-point 계산:
# 용량이 0.95 Ah (공칭 1.1Ah 대비 약 86%) 이하로 처음 떨어지는 시점 또는
# Cycle 100 이후 기울기 변화

# 시각화: 2x2 플롯
# 1. Batch 1 전체 셀 열화 곡선 (장수명 vs 단수명 색상 구분)
# 2. 초기 100사이클 확대 (용량 변화 미미함 입증)
# 3. 대표 셀 비교 (최단수명, 중앙값, 최장수명)
# 4. 사이클당 열화율 (dQ/dCycle) 추이

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# EOL 기준선 (1.1 Ah의 80% = 0.88 Ah)
eol_val = 0.88

# 1. Batch 1 전체 셀 방전용량 곡선
ax1 = axes[0, 0]
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(df_b1_clean["cycle_life"].min(), df_b1_clean["cycle_life"].max())

for _, r in df_b1_clean.iterrows():
    qd = r["QDischarge"][1:] # Cycle 2부터
    cycles = np.arange(2, len(qd) + 2)
    # EOL 도달 시점까지만 플롯 (또는 전체)
    color = cmap(norm(r["cycle_life"]))
    ax1.plot(cycles, qd, color=color, alpha=0.6, linewidth=1.2)

ax1.axhline(eol_val, color="red", linestyle="--", linewidth=1.5, label="EOL (0.88 Ah = 80%)")
ax1.axvline(100, color="gray", linestyle=":", linewidth=1.5, label="예측 시점 (Cycle 100)")
ax1.set_title("Batch 1 전체 셀 방전용량 열화 곡선 (색상: 수명)")
ax1.set_xlabel("사이클 수 (Cycle)")
ax1.set_ylabel("방전용량 (Ah)")
ax1.set_xlim(0, 1300)
ax1.set_ylim(0.75, 1.15)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower left")

# 컬러바 추가
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1)
cbar.set_label("수명 (Cycle Life)")

# 2. 초기 100사이클 확대도
ax2 = axes[0, 1]
for _, r in df_b1_clean.iterrows():
    qd = r["QDischarge"][1:100] # Cycle 2 ~ 100
    cycles = np.arange(2, len(qd) + 2)
    color = cmap(norm(r["cycle_life"]))
    ax2.plot(cycles, qd, color=color, alpha=0.6, linewidth=1.2)

ax2.set_title("초기 100사이클 구간 확대 (Cycle 2 ~ 100)")
ax2.set_xlabel("사이클 수 (Cycle)")
ax2.set_ylabel("방전용량 (Ah)")
ax2.set_xlim(2, 100)
ax2.set_ylim(0.95, 1.12)
ax2.grid(True, linestyle="--", alpha=0.5)

# 3. 대표 셀 비교 (최단수명, 중앙값, 최장수명)
ax3 = axes[1, 0]
min_cell = df_b1_clean.loc[df_b1_clean["cycle_life"].idxmin()]
med_idx = (df_b1_clean["cycle_life"] - df_b1_clean["cycle_life"].median()).abs().idxmin()
med_cell = df_b1_clean.loc[med_idx]
max_cell = df_b1_clean.loc[df_b1_clean["cycle_life"].idxmax()]

rep_cells = [
    ("최단수명 (534c)", min_cell, "#d9534f"),
    ("중앙값 (859c)", med_cell, "#f0ad4e"),
    ("최장수명 (1227c)", max_cell, "#5cb85c")
]

for label, cell, col in rep_cells:
    qd = cell["QDischarge"][1:]
    cycles = np.arange(2, len(qd) + 2)
    ax3.plot(cycles, qd, color=col, linewidth=2.0, label=f"{label}: {cell['cell_id']}")

ax3.axhline(eol_val, color="red", linestyle="--", linewidth=1.2, label="EOL (0.88 Ah)")
ax3.axvline(100, color="gray", linestyle=":", linewidth=1.2, label="Cycle 100")
ax3.set_title("수명 구간별 대표 셀 열화 궤적 비교")
ax3.set_xlabel("사이클 수 (Cycle)")
ax3.set_ylabel("방전용량 (Ah)")
ax3.set_xlim(0, 1300)
ax3.set_ylim(0.75, 1.15)
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.legend(loc="lower left")

# 4. 열화 가속도(Knee-point) 분석: 이동 평균 기울기 dQ/dN
ax4 = axes[1, 1]
for label, cell, col in rep_cells:
    qd = pd.Series(cell["QDischarge"][1:])
    # 20 사이클 롤링 기울기 (Ah / cycle)
    rolling_slope = (qd.diff(20) / 20).rolling(10).mean() * 1000 # mAh / cycle
    cycles = np.arange(2, len(qd) + 2)
    ax4.plot(cycles, rolling_slope, color=col, linewidth=1.8, label=label)

ax4.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
ax4.axvline(100, color="gray", linestyle=":", linewidth=1.2, label="Cycle 100")
ax4.set_title("열화 속도 추이 (20사이클 롤링 dQ/dN, 단위: mAh/cycle)")
ax4.set_xlabel("사이클 수 (Cycle)")
ax4.set_ylabel("열화 속도 (mAh / cycle)")
ax4.set_xlim(0, 1300)
ax4.set_ylim(-3.0, 0.5)
ax4.grid(True, linestyle="--", alpha=0.5)
ax4.legend(loc="lower left")

fig.tight_layout()
fig_path = os.path.join(results_dir, "q2_discharge_capacity_degradation.png")
fig.savefig(fig_path, dpi=160)
plt.close(fig)
print(f"Figure saved to {fig_path}")
