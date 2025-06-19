# nn.py
# encoding: utf-8

import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from PyEMD.EMD import EMD
from scipy.signal import hilbert
from sklearn.preprocessing import RobustScaler
from itertools import groupby
from operator import itemgetter

def extract_and_aggregate(eventalign_path):
    """
    Aggregate raw signal data from eventalign.txt
    """
    merged_data = defaultdict(lambda: {
        'count': 0, 'col4_sum': 0, 'col5_sum': 0, 'col6_sum': 0, 'col7': []
    })

    with open(eventalign_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) != 16:
                continue
            contig = fields[0].split('.')[0]
            key = (contig, fields[1], fields[2], fields[3])
            col4 = float(fields[6])
            col5 = float(fields[7])
            col6 = float(fields[8])
            col7 = fields[15].split(',')

            merged_data[key]['count'] += 1
            merged_data[key]['col4_sum'] += col4
            merged_data[key]['col5_sum'] += col5
            merged_data[key]['col6_sum'] += col6
            merged_data[key]['col7'].extend(col7)

    rows = []
    for key, values in merged_data.items():
        avg_col4 = values['col4_sum'] / values['count']
        avg_col5 = values['col5_sum'] / values['count']
        avg_col6 = values['col6_sum'] / values['count']
        merged_col7 = ','.join(values['col7'])
        row = list(key) + [f"{avg_col4:.2f}", f"{avg_col5:.3f}", f"{avg_col6:.5f}", merged_col7]
        rows.append(row)

    return rows

def compute_imf_features(rows):
    """
    Apply EMD and Hilbert transform to extract IMF features
    """
    output_rows = []
    rows_sorted = sorted(rows, key=lambda x: x[3])

    for key, group in groupby(rows_sorted, key=itemgetter(3)):
        group_list = list(group)
        for i in range(len(group_list)):
            row = group_list[i]
            signal = [float(x) for x in row[7].split(',')]
            max_val = np.max(signal)
            min_val = np.min(signal)
            median_val = np.median(signal)

            # Merge nearby signal values with sliding window
            merged_signal = []
            for j in range(i - 4, i + 5):
                if 0 <= j < len(group_list):
                    merged_signal.extend([float(x) for x in group_list[j][7].split(',')])
            merged_values = np.array(merged_signal)

            max_amps = [max_val, min_val, median_val]

            emd = EMD()
            imfs = emd(merged_values)

            for imf in reversed(imfs):
                amplitude = np.abs(hilbert(imf))
                max_amp = np.max(amplitude)
                max_amps.append(max_amp)
                if len(max_amps) == 12:
                    break

            if len(max_amps) < 12:
                max_amps.extend([0] * (12 - len(max_amps)))

            output_rows.append(row[:7] + [f"{float(a):.4f}" for a in max_amps])

    return output_rows

def nn_encoding_and_windowing(rows):
    """
    Apply NN encoding and sliding window feature extension
    """
    df_all = pd.DataFrame(rows)
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
                'C': df2.iloc[:, i].shift(4),
                'D': df2.iloc[:, i].shift(3),
                'E': df2.iloc[:, i].shift(2),
                'F': df2.iloc[:, i].shift(1),
                'H': df2.iloc[:, i].shift(-1),
                'I': df2.iloc[:, i].shift(-2),
                'J': df2.iloc[:, i].shift(-3),
                'K': df2.iloc[:, i].shift(-4)
            })
            lagged_dfs.append(lagged_df)

        df_lagged_combined = pd.concat(lagged_dfs, axis=1)
        df3 = pd.concat([df2, df_lagged_combined], axis=1)
        df3 = df3.iloc[4:-4].copy()
        df3.fillna(0, inplace=True)
        all_results.append(df3)

    return pd.concat(all_results, axis=0).reset_index(drop=True)

def scale_and_filter(df):
    """
    Scale feature columns with RobustScaler and filter A-sites
    """
    result_df = pd.DataFrame()
    grouped = df.groupby(df.columns[3], group_keys=False)

    for _, group in grouped:
        group = group.copy()    
        group.iloc[:, 1] = group.iloc[:, 1].astype(int) + 3

        feature_start_idx = 40
        features = group.iloc[:, feature_start_idx:].values
        scaler = RobustScaler()
        scaled = scaler.fit_transform(features)
        scaled = np.round(scaled, 4)

        group.iloc[:, feature_start_idx:] = scaled

        # Filter rows where the 5th character in the 3rd column is 'A'
        filtered_group = group[group.iloc[:, 2].astype(str).str[4] == 'A']
        result_df = pd.concat([result_df, filtered_group], ignore_index=True)

    return result_df

def main():
    parser = argparse.ArgumentParser(description="Preprocess eventalign data to final training format")
    parser.add_argument('--eventalign', required=True, help='Path to eventalign.txt')
    parser.add_argument('--out_dir', required=True, help='Directory to save final result')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("🧬 Step 1: Aggregating raw signal...")
    rows = extract_and_aggregate(args.eventalign)

    print("📈 Step 2: Extracting EMD + Hilbert features...")
    imf_rows = compute_imf_features(rows)

    print("🔤 Step 3: NN encoding and sliding window generation...")
    df = nn_encoding_and_windowing(imf_rows)

    print("🧪 Step 4: Scaling and filtering...")
    df_result = scale_and_filter(df)
    df_result.columns = ['config', 'position', 'motif', 'read_index'] + [f'one_hot{i}' for i in range(1, 37)] + [
        'mean','stdv','length','max','min','medium',
        'max_amp1','max_amp2','max_amp3','max_amp4','max_amp5','max_amp6','max_amp7','max_amp8','max_amp9',
        'mean-4', 'mean-3', 'mean-2', 'mean-1', 'mean+1', 'mean+2', 'mean+3', 'mean+4',
        'std-4', 'std-3', 'std-2', 'std-1', 'std+1', 'std+2', 'std+3', 'std+4',
        'length-4', 'length-3', 'length-2', 'length-1', 'length+1', 'length+2', 'length+3', 'length+4',
        'max-4', 'max-3', 'max-2', 'max-1', 'max+1', 'max+2', 'max+3', 'max+4',
        'min-4', 'min-3', 'min-2', 'min-1', 'min+1', 'min+2', 'min+3', 'min+4',
        'medium-4', 'medium-3', 'medium-2', 'medium-1', 'medium+1', 'medium+2', 'medium+3', 'medium+4'
    ]

    columns_order = ['config', 'position', 'motif', 'read_index'] + [f'one_hot{i}' for i in range(1, 37)] + [
        'mean-4', 'mean-3', 'mean-2', 'mean-1', 'mean', 'mean+1', 'mean+2', 'mean+3', 'mean+4',
        'std-4', 'std-3', 'std-2', 'std-1', 'stdv', 'std+1', 'std+2', 'std+3', 'std+4',
        'length-4', 'length-3', 'length-2', 'length-1', 'length', 'length+1', 'length+2', 'length+3', 'length+4',
        'max-4', 'max-3', 'max-2', 'max-1', 'max', 'max+1', 'max+2', 'max+3', 'max+4',
        'min-4', 'min-3', 'min-2', 'min-1', 'min', 'min+1', 'min+2', 'min+3', 'min+4',
        'medium-4', 'medium-3', 'medium-2', 'medium-1', 'medium', 'medium+1', 'medium+2', 'medium+3', 'medium+4',
        'max_amp1','max_amp2','max_amp3','max_amp4','max_amp5','max_amp6','max_amp7','max_amp8','max_amp9'
    ]
    df_result = df_result.reindex(columns=columns_order)
    out_path = os.path.join(args.out_dir, 'data_feature.txt')
    df_result.to_csv(out_path, index=False, sep='\t', float_format='%.4f')
    print(f"✅ Done! Data feature saved to: {out_path}")

if __name__ == '__main__':
    main()
