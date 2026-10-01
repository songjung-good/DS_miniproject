# ponytail: minimal h5py loader for MIT-Stanford battery mat files. Upgrades: add cycle-level detail extraction if needed.
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
            # cycle_life
            ref_life = batch['cycle_life'][i, 0]
            life = float(np.array(f[ref_life])[0, 0])
            
            # policy
            ref_pol = batch['policy_readable'][i, 0]
            policy = ''.join(chr(c) for c in np.array(f[ref_pol]).flatten())
            
            # channel_id
            ref_ch = batch['channel_id'][i, 0]
            channel_id = int(np.array(f[ref_ch])[0, 0])
            
            # barcode
            ref_bar = batch['barcode'][i, 0]
            barcode = '-'.join(map(str, np.array(f[ref_bar]).flatten())) if ref_bar else ""
            
            # summary fields
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
