import nbformat as nbf
import os

project_root = "/Users/skala_yh/DS_miniproject"
nb_path = os.path.join(project_root, "notebooks", "01_EDA.ipynb")

nb = nbf.read(nb_path, as_version=4)

# Section 9 Markdown
q2_md_intro = nbf.v4.new_markdown_cell("""## 9. Q2 방전 용량 열화 곡선 및 Knee-point 분석

- **질문**: 방전 용량은 어떻게 감소하는가? (열화 곡선)
- **확인할 것**: 사이클별 $Q_d$ 추이, 열화 속도가 일정한지 가속되는지(비선형성), Knee-point(급격한 열화 시작점) 탐색.
- **분석 범위**: Batch 1 클린 셀(46개) 대상 Cycle 2부터 수명 종료(EOL, 0.88 Ah = 공칭 1.1 Ah의 80%)까지의 열화 궤적을 분석합니다.
""")

# Section 9 Code Cell
q2_code = nbf.v4.new_code_cell("""# 1. 초기 100사이클 vs 전체 수명 열화 통계 집계
q2_vals = [r["QDischarge"][1] for _, r in df_b1.iterrows()]
q100_vals = [r["QDischarge"][99] for _, r in df_b1.iterrows()]
delta_q100_2 = np.array(q100_vals) - np.array(q2_vals)

knee_cycles = []
for _, r in df_b1.iterrows():
    qd = r["QDischarge"][1:]
    cycles = np.arange(2, len(qd)+2)
    below_95 = np.where(qd <= 0.95)[0]
    knee_c = cycles[below_95[0]] if len(below_95) > 0 else r["cycle_life"]
    knee_cycles.append(knee_c)

knee_cycles = np.array(knee_cycles)
cycles_to_eol = df_b1["cycle_life"].values - knee_cycles

q2_stat_df = pd.DataFrame([{
    "Q2 평균 (Ah)": np.mean(q2_vals),
    "Q100 평균 (Ah)": np.mean(q100_vals),
    "초기 용량변화 (Q100-Q2) (mAh)": np.mean(delta_q100_2) * 1000,
    "초기 용량증가 셀 비율 (%)": (delta_q100_2 >= 0).mean() * 100,
    "초기 열화율 (mAh/cycle)": (np.mean(delta_q100_2) / 98) * 1000,
    "전체 평균 열화율 (mAh/cycle)": np.mean([(0.88 - r["QDischarge"][1]) / (r["cycle_life"] - 2) * 1000 for _, r in df_b1.iterrows()]),
    "평균 Knee 사이클 (0.95Ah)": np.mean(knee_cycles),
    "Knee 후 EOL까지 남은 사이클": np.mean(cycles_to_eol),
    "Knee 후 EOL 도달 비율 (%)": (np.mean(cycles_to_eol) / df_b1["cycle_life"].mean()) * 100
}])
display(q2_stat_df.style.format("{:.2f}"))

# 2. 2x2 열화 곡선 및 Knee-point 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
eol_val = 0.88

# (1) 전체 셀 열화 곡선
ax1 = axes[0, 0]
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(df_b1["cycle_life"].min(), df_b1["cycle_life"].max())

for _, r in df_b1.iterrows():
    qd = r["QDischarge"][1:]
    cycles = np.arange(2, len(qd) + 2)
    color = cmap(norm(r["cycle_life"]))
    ax1.plot(cycles, qd, color=color, alpha=0.5, linewidth=1.1)

ax1.axhline(eol_val, color="red", linestyle="--", linewidth=1.5, label="EOL (0.88 Ah = 80%)")
ax1.axvline(100, color="gray", linestyle=":", linewidth=1.5, label="조기 예측 시점 (Cycle 100)")
ax1.set_title("Batch 1 전체 셀 방전용량 열화 궤적 (색상: 수명)")
ax1.set_xlabel("사이클 수 (Cycle)")
ax1.set_ylabel("방전용량 (Ah)")
ax1.set_xlim(0, 1300)
ax1.set_ylim(0.75, 1.15)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower left")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1)
cbar.set_label("수명 (Cycle Life)")

# (2) 초기 100사이클 확대
ax2 = axes[0, 1]
for _, r in df_b1.iterrows():
    qd = r["QDischarge"][1:100]
    cycles = np.arange(2, len(qd) + 2)
    color = cmap(norm(r["cycle_life"]))
    ax2.plot(cycles, qd, color=color, alpha=0.5, linewidth=1.1)

ax2.set_title("초기 100사이클 구간 확대 (Cycle 2 ~ 100)")
ax2.set_xlabel("사이클 수 (Cycle)")
ax2.set_ylabel("방전용량 (Ah)")
ax2.set_xlim(2, 100)
ax2.set_ylim(0.95, 1.12)
ax2.grid(True, linestyle="--", alpha=0.5)

# (3) 대표 셀 열화 비교
ax3 = axes[1, 0]
min_cell = df_b1.loc[df_b1["cycle_life"].idxmin()]
med_idx = (df_b1["cycle_life"] - df_b1["cycle_life"].median()).abs().idxmin()
med_cell = df_b1.loc[med_idx]
max_cell = df_b1.loc[df_b1["cycle_life"].idxmax()]

rep_cells = [
    (f"최단수명 ({min_cell['cycle_life']:.0f}c)", min_cell, "#d9534f"),
    (f"중앙값 ({med_cell['cycle_life']:.0f}c)", med_cell, "#f0ad4e"),
    (f"최장수명 ({max_cell['cycle_life']:.0f}c)", max_cell, "#5cb85c")
]

for label, cell, col in rep_cells:
    qd = cell["QDischarge"][1:]
    cycles = np.arange(2, len(qd) + 2)
    ax3.plot(cycles, qd, color=col, linewidth=2.0, label=f"{label}: {cell['cell_id']}")

ax3.axhline(eol_val, color="red", linestyle="--", linewidth=1.2, label="EOL (0.88 Ah)")
ax3.axvline(100, color="gray", linestyle=":", linewidth=1.2, label="Cycle 100")
ax3.set_title("대표 셀(최단·중앙·최장) 열화 궤적 비교")
ax3.set_xlabel("사이클 수 (Cycle)")
ax3.set_ylabel("방전용량 (Ah)")
ax3.set_xlim(0, 1300)
ax3.set_ylim(0.75, 1.15)
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.legend(loc="lower left")

# (4) 열화 가속도 (20사이클 롤링 기울기 dQ/dN)
ax4 = axes[1, 1]
for label, cell, col in rep_cells:
    qd = pd.Series(cell["QDischarge"][1:])
    rolling_slope = (qd.diff(20) / 20).rolling(10).mean() * 1000
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
fig_path = os.path.join(project_root, "results", "q2_discharge_capacity_degradation.png")
fig.savefig(fig_path, dpi=160)
plt.show()
""")

# Section 9 Observations Markdown
q2_md_obs = nbf.v4.new_markdown_cell("""### Q2 관측과 생각

1. **비선형 2단계 가속 열화 실증**:
   - 배터리 열화는 선형적으로 서서히 감소하지 않으며, **완만한 안정기(초기~중기)**를 거쳐 특정 임계점 이후 수직으로 급락하는 **급격 열화기(Knee-point 이후)**의 전형적인 2단계 비선형 거동을 보입니다.
2. **초기 100사이클의 화성 안정화(Break-in) 현상**:
   - 초기 100사이클 구간에서 용량이 감소하기는커녕, Batch 1 셀의 **91.3%(42/46개)**에서 오히려 용량이 미세하게 증가했습니다 ($Q_2$ 평균 1.078 Ah $\\rightarrow$ $Q_{100}$ 평균 1.081 Ah, 평균 $+2.60\\text{ mAh}$).
   - 초기 100사이클 열화율은 $+0.027\\text{ mAh/cycle}$인 반면, 전체 수명 열화율은 $-0.246\\text{ mAh/cycle}$로 정반대 거동을 보입니다.
3. **Knee-point 도달 후 급격한 종말**:
   - 용량이 0.95 Ah(공칭 대비 ~86%)로 꺾이는 Knee-point는 평균 **785.6사이클**에 도달합니다.
   - Knee-point에 진입한 이후 EOL(0.88 Ah = 80%)까지 도달하는 데 걸리는 시간은 평균 **59.1사이클(전체 수명의 약 7.0%)**에 불과합니다. 임계점을 넘으면 불과 수십 사이클 만에 배터리가 급사합니다.
4. **피처 엔지니어링 및 모델링 시사점**:
   - **선형 외삽 모델 완전 배제**: 초기 100사이클의 단순 기울기를 외삽하면 용량이 증가하므로 수명이 무한대로 예측되는 치명적 오류가 발생합니다.
   - **스칼라 용량의 무용성**: 초기 100사이클 확대 그래프에서 보듯, 장수명 셀과 단수명 셀의 초기 용량 궤적이 완전히 겹쳐 구분할 수 없습니다.
   - **미세 구조적 신호($\\Delta Q(V)$)의 절대적 필요성**: 겉보기 방전용량($Q_d$)으로는 보이지 않는 전극 활물질 손실을 감지하기 위해 Q3의 전압별 용량 변화 곡선($\\Delta Q_{100-10}(V)$) 분석이 수명 예측의 필수 관건임을 입증합니다.
""")

nb.cells.extend([q2_md_intro, q2_code, q2_md_obs])
nbf.write(nb, nb_path)
print("Successfully appended Q2 to 01_EDA.ipynb")
