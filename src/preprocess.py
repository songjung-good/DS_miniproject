# ponytail: clean data loader and preprocessing module for MIT-Stanford battery dataset.
import os
import h5py
import numpy as np
import pandas as pd

def load_batch_summary(mat_path, batch_name=""):
    """
    MATLAB v7.3 .mat 파일에서 셀 단위 요약 데이터와 메타데이터를 DataFrame으로 로드.
    """
    records = []
    with h5py.File(mat_path, 'r') as f:
        batch = f['batch']
        n_cells = batch['cycle_life'].shape[0]
        
        for i in range(n_cells):
            ref_life = batch['cycle_life'][i, 0]
            life = float(np.array(f[ref_life])[0, 0])
            
            ref_pol = batch['policy_readable'][i, 0]
            policy = ''.join(chr(c) for c in np.array(f[ref_pol]).flatten())
            
            ref_ch = batch['channel_id'][i, 0]
            channel_id = int(np.array(f[ref_ch])[0, 0])
            
            ref_bar = batch['barcode'][i, 0]
            barcode = '-'.join(map(str, np.array(f[ref_bar]).flatten())) if ref_bar else ""
            
            ref_sum = batch['summary'][i, 0]
            sum_grp = f[ref_sum]
            
            q_dis = np.array(sum_grp['QDischarge']).flatten()
            q_ch = np.array(sum_grp['QCharge']).flatten()
            ir = np.array(sum_grp['IR']).flatten()
            tavg = np.array(sum_grp['Tavg']).flatten()
            tmax = np.array(sum_grp['Tmax']).flatten()
            tmin = np.array(sum_grp['Tmin']).flatten()
            ch_time = np.array(sum_grp['chargetime']).flatten()
            cycles = np.array(sum_grp['cycle']).flatten()
            
            records.append({
                'cell_id': f"{batch_name}_c{i:02d}",
                'cell_idx': i,
                'batch': batch_name,
                'channel_id': channel_id,
                'barcode': barcode,
                'policy': policy,
                'cycle_life': life,
                'n_cycles': len(cycles),
                'cycles': cycles,
                'QDischarge': q_dis,
                'QCharge': q_ch,
                'IR': ir,
                'Tavg': tavg,
                'Tmax': tmax,
                'Tmin': tmin,
                'chargetime': ch_time
            })
            
    return pd.DataFrame(records)

def filter_clean_cells(df):
    """
    유한한 양수 수명 라벨이 있는 셀과 없는 셀을 분리한다.
    실험상 결측 원인이나 최종 피처 유효성은 여기서 판정하지 않는다.
    """
    valid_mask = np.isfinite(df['cycle_life']) & (df['cycle_life'] > 0)
    valid_df = df[valid_mask].copy().reset_index(drop=True)
    excluded_df = df[~valid_mask].copy().reset_index(drop=True)
    return valid_df, excluded_df

def extract_clean_features(df, max_cycle=100):
    """
    정제된 셀에 대해 Cycle 2 ~ max_cycle (기본 100) 초기 물리 피처 추출.
    - 주의: 타깃(cycle_life)과 사후정보(n_cycles)는 피처 세트에서 분리 관리.
    - 슬라이스 [1:max_cycle]을 적용하여 Cycle 1의 초기화 0.0을 배제.
    """
    rows = []
    for _, r in df.iterrows():
        qd = r['QDischarge']
        qc = r['QCharge']
        ir = r['IR']
        tmax = r['Tmax']
        tavg = r['Tavg']
        tmin = r['Tmin']
        ch_time = r['chargetime']
        
        # Cycle 2 (index 1) & Cycle 100 (index max_cycle - 1)
        q_init = qd[1] if len(qd) > 1 else np.nan
        qc_init = qc[1] if len(qc) > 1 else np.nan
        ce_init = (q_init / qc_init * 100) if (qc_init and not np.isnan(qc_init)) else np.nan
        
        end_idx = max_cycle - 1
        q_end = qd[end_idx] if len(qd) > end_idx else np.nan
        qc_end = qc[end_idx] if len(qc) > end_idx else np.nan
        ce_end = (q_end / qc_end * 100) if (qc_end and not np.isnan(qc_end)) else np.nan
        delta_q = (q_end - q_init) if (not np.isnan(q_end) and not np.isnan(q_init)) else np.nan
        
        ir_init = ir[1] if len(ir) > 1 else np.nan
        ir_end = ir[end_idx] if len(ir) > end_idx else np.nan
        delta_ir = (ir_end - ir_init) if (not np.isnan(ir_end) and not np.isnan(ir_init)) else np.nan
        
        # Cycle 2 to max_cycle (index 1 to max_cycle)
        t_min = np.min(tmin[1:max_cycle]) if len(tmin) >= max_cycle else np.nan
        t_avg = np.mean(tavg[1:max_cycle]) if len(tavg) >= max_cycle else np.nan
        t_max = np.max(tmax[1:max_cycle]) if len(tmax) >= max_cycle else np.nan
        delta_t = (t_max - t_min) if (not np.isnan(t_max) and not np.isnan(t_min)) else np.nan
        
        ch_init = ch_time[1] if len(ch_time) > 1 else np.nan
        ch_end = ch_time[end_idx] if len(ch_time) > end_idx else np.nan
        ch_avg = np.mean(ch_time[1:max_cycle]) if len(ch_time) >= max_cycle else np.nan
        
        rows.append({
            'cell_id': r['cell_id'],
            'batch': r['batch'],
            'policy': r['policy'],
            'cycle_life': r['cycle_life'], # Target
            'n_cycles': r['n_cycles'],     # Post-experimental count
            # Input Feature Candidates (X)
            'q_init_c2': q_init,
            'qc_init_c2': qc_init,
            'ce_init_pct': ce_init,
            'q_c100': q_end,
            'qc_c100': qc_end,
            'ce_c100_pct': ce_end,
            'delta_q_100_2': delta_q,
            'ir_init': ir_init,
            'ir_c100': ir_end,
            'delta_ir_100_2': delta_ir,
            't_min_100': t_min,
            't_avg_100': t_avg,
            't_max_100': t_max,
            'delta_t_100': delta_t,
            'ch_time_init_min': ch_init,
            'ch_time_c100_min': ch_end,
            'ch_time_avg_min': ch_avg
        })
    return pd.DataFrame(rows)
