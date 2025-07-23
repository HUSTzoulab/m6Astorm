# m6astorm_dataprep.py

import os
import argparse
import pandas as pd
import numpy as np
from multiprocessing import Pool
from collections import defaultdict
from joblib import Parallel, delayed
from PyEMD import EMD
from scipy.signal import hilbert
from itertools import groupby
from sklearn.preprocessing import RobustScaler
from operator import itemgetter

### ---------------- STEP 1 ----------------

def parse_eventalign(eventalign_path, chunk_size=2000000, out_dir="./tmp_chunks"):
    os.makedirs(out_dir, exist_ok=True)

    def flush_to_file(batch_data, idx):
        out_file = os.path.join(out_dir, f"chunk_{idx:04d}.csv")
        pd.DataFrame(batch_data).to_csv(out_file, index=False, sep='\t',header=None)

    merged_data = defaultdict(lambda: {'count': 0, 'col4': 0, 'col5': 0, 'col6': 0, 'col7': []})
    current_batch = []
    chunk_idx = 0

    with open(eventalign_path, 'r') as f:
        next(f)
        for line_idx, line in enumerate(f, 1):
            fields = line.strip().split('\t')
            if len(fields) != 16:
                continue
            contig = fields[0].split('.')[0]
            key = (contig, fields[1], fields[2], fields[3])
            col4 = float(fields[6])
            col5 = float(fields[7])
            col6 = float(fields[8])
            col7 = fields[15].split(',')

            md = merged_data[key]
            md['count'] += 1
            md['col4'] += col4
            md['col5'] += col5
            md['col6'] += col6
            md['col7'].extend(col7)

            if line_idx % chunk_size == 0:
                for k, v in merged_data.items():
                    avg4 = float(f"{v['col4'] / v['count']:.6f}")
                    avg5 = float(f"{v['col5'] / v['count']:.6f}")
                    avg6 = float(f"{v['col6'] / v['count']:.6f}")
                    merged_col7 = ','.join(v['col7'])
                    current_batch.append(list(k) + [avg4, avg5, avg6, merged_col7])
                flush_to_file(current_batch, chunk_idx)
                current_batch.clear()
                merged_data.clear()
                chunk_idx += 1

    if merged_data:
        for k, v in merged_data.items():
            avg4 = float(f"{v['col4'] / v['count']:.6f}")
            avg5 = float(f"{v['col5'] / v['count']:.6f}")
            avg6 = float(f"{v['col6'] / v['count']:.6f}")
            merged_col7 = ','.join(v['col7'])
            current_batch.append(list(k) + [avg4, avg5, avg6, merged_col7])
        flush_to_file(current_batch, chunk_idx)


### ---------------- STEP 2 ----------------
def process_imf_chunk(chunk_file, output_file):
    try:
        df = pd.read_csv(chunk_file, sep='\t', header=None, low_memory=False)
    except Exception as e:
        print(f"[ERROR] Failed to load {chunk_file}: {e}")
        return

    rows = df.itertuples(index=False, name=None)
    rows_sorted = sorted(rows, key=itemgetter(3))  

    output_rows = []

    for read_id, group in groupby(rows_sorted, key=itemgetter(3)):
        group_list = list(group)
        group_len = len(group_list)

        for i, row in enumerate(group_list):
            try:
                signal = [float(x) for x in row[7].split(',')]
            except Exception:
                continue

            max_val = np.max(signal)
            min_val = np.min(signal)
            median_val = np.median(signal)

            merged_signal = []
            for j in range(max(0, i - 4), min(group_len, i + 5)):
                try:
                    merged_signal.extend([float(x) for x in group_list[j][7].split(',')])
                except Exception:
                    continue

            merged_values = np.array(merged_signal, dtype=np.float32)
            max_amps = [max_val, min_val, median_val]

            try:
                emd = EMD()
                imfs = emd(merged_values)
                for imf in reversed(imfs):
                    amplitude = np.abs(hilbert(imf))
                    max_amp = np.max(amplitude)
                    max_amps.append(max_amp)
                    if len(max_amps) == 12:
                        break
            except Exception as e:
                warnings.warn(f"[WARN] EMD failed for row {i} in {chunk_file}: {e}")
                continue
           
            if len(max_amps) < 12:
                max_amps.extend([0.0] * (12 - len(max_amps)))
            
            output_rows.append(row[:7] + tuple(f"{a:.4f}" for a in max_amps))

    pd.DataFrame(output_rows).to_csv(output_file, index=False, sep='\t', header=False)


### ---------------- STEP 3 ----------------
def process_window_chunk(input_file, output_file):
    """
    Apply NN encoding, one-hot encoding, and sliding window features on a single CSV file
    """
    print(f"[3] Processing {input_file}...")

    df_all = pd.read_csv(input_file, sep='\t', header=None)
    group_key = df_all.iloc[:, 3]
    grouped = df_all.groupby(group_key, group_keys=False)
    all_results = []

    for _, df in grouped:
        df = df.reset_index(drop=True)
        df1 = df.iloc[:, 0:4].copy()
        new_column = []

        for i in range(len(df1)):
            if i < 2 or i >= len(df1) - 2:
                new_value = "NN" + df1.iloc[i, 2] + "NN"
            else:
                pre2 = df1.iloc[i - 2, 2][:2]
                now5 = df1.iloc[i, 2][:5]
                suf2 = df1.iloc[i + 2, 2][-2:]
                new_value = pre2 + now5 + suf2
            new_column.append(new_value)

        df1.iloc[:, 2] = new_column

        # One-hot encoding
        new_cols = [f'column{i}' for i in range(1, 10)]
        df1[new_cols] = df1.iloc[:, 2].apply(lambda x: pd.Series(list(x)))
        encoding_dict = {'A': '1,0,0,0', 'T': '0,1,0,0', 'G': '0,0,1,0', 'C': '0,0,0,1', 'N': '0,0,0,0'}
        for col in new_cols:
            df1[col] = df1[col].replace(encoding_dict, regex=True)
        for i in range(1, 10):
            df1[[f'column{i}_1', f'column{i}_2', f'column{i}_3', f'column{i}_4']] = df1[f'column{i}'].str.split(',', expand=True)
        df1.drop(columns=new_cols, inplace=True)

        df2 = pd.concat([df1, df.iloc[:, 4:].reset_index(drop=True)], axis=1)

        # Create sliding window features
        lagged_dfs = []
        for i in range(40, 46):
            lagged_df = pd.DataFrame({
                f'{i}_C': df2.iloc[:, i].shift(4),
                f'{i}_D': df2.iloc[:, i].shift(3),
                f'{i}_E': df2.iloc[:, i].shift(2),
                f'{i}_F': df2.iloc[:, i].shift(1),
                f'{i}_H': df2.iloc[:, i].shift(-1),
                f'{i}_I': df2.iloc[:, i].shift(-2),
                f'{i}_J': df2.iloc[:, i].shift(-3),
                f'{i}_K': df2.iloc[:, i].shift(-4)
            })
            lagged_dfs.append(lagged_df)

        df_lagged_combined = pd.concat(lagged_dfs, axis=1)
        df3 = pd.concat([df2, df_lagged_combined], axis=1)
        df3 = df3.iloc[4:-4].copy()
        df3.fillna(0, inplace=True)
        all_results.append(df3)

    final_df = pd.concat(all_results, axis=0).reset_index(drop=True)
    final_df.to_csv(output_file, sep='\t', index=False, header=False)


### ---------------- STEP 4 ----------------
def scale_and_filter_file(input_path, output_path):
    """
    Load one chunk CSV, scale numerical features, and filter windows with 'A' at center.
    """
    df = pd.read_csv(input_path, sep='\t', header=None)
    result_df = pd.DataFrame()
    grouped = df.groupby(df.columns[3], group_keys=False)  # group by read_index

    for _, group in grouped:
        group = group.copy()
        group.iloc[:, 1] = group.iloc[:, 1].astype(int) + 3  # adjust center position

        feature_start_idx = 40
        features = group.iloc[:, feature_start_idx:].values

        scaler = RobustScaler()
        scaled = scaler.fit_transform(features)
        scaled = np.round(scaled, 4)
        group.iloc[:, feature_start_idx:] = scaled

        # Filter rows where the 5th character in the 3rd column (sequence window) is 'A'
        filtered_group = group[group.iloc[:, 2].astype(str).str[4] == 'A']
        result_df = pd.concat([result_df, filtered_group], ignore_index=True)

    result_df.to_csv(output_path, index=False, sep='\t', header=None)


### ---------------- MAIN ----------------
def run_all(eventalign, out_dir, n_jobs=48):
    os.makedirs(out_dir, exist_ok=True)

    chunk_dir = os.path.join(out_dir, "tmp_chunks")
    imf_dir = os.path.join(out_dir, "tmp_imf")
    windowed_dir = os.path.join(out_dir, "tmp_windowed")
    scaled_dir = os.path.join(out_dir, "tmp_scaled")

    print("[1] Parsing eventalign file in chunks...")
    parse_eventalign(eventalign, out_dir=chunk_dir)

    print("[2] Computing IMF features...")
    os.makedirs(imf_dir, exist_ok=True)
    chunk_files = sorted(os.listdir(chunk_dir))
    Parallel(n_jobs=n_jobs)(
        delayed(process_imf_chunk)(os.path.join(chunk_dir, f), os.path.join(imf_dir, f))
        for f in chunk_files
    )

    print("[3] Windowing + One-hot encoding...")
    os.makedirs(windowed_dir, exist_ok=True)
    chunk_files = sorted(os.listdir(imf_dir))
    Parallel(n_jobs=n_jobs)(
        delayed(process_window_chunk)(os.path.join(imf_dir, f), os.path.join(windowed_dir, f))
        for f in chunk_files
    )

    print("[4] Scaling and filtering...")
    os.makedirs(scaled_dir, exist_ok=True)
    chunk_files = sorted(os.listdir(windowed_dir))
    Parallel(n_jobs=n_jobs)(
        delayed(scale_and_filter_file)(os.path.join(windowed_dir, f), os.path.join(scaled_dir, f))
        for f in chunk_files
    )

    print(f"✅ Done! Data feature saved to: {combined_csv}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="m6Astorm full data pre-processing pipeline")
    parser.add_argument('--eventalign', required=True, help='Input eventalign.txt file (TSV)')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--n_jobs', type=int, default=48, help='Number of parallel workers')
    args = parser.parse_args()

    run_all(args.eventalign, args.out_dir, args.n_jobs)