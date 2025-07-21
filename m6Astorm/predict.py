import os
import sys
import argparse
import warnings
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import nn
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pre', type=str, required=True, help='Path to preprocessed data')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--n_jobs', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--min_coverage', type=int, default=20, help='Minimum coverage threshold')
    parser.add_argument('--mod_prob_thresh', type=float, default=0.5, help='Modification probability threshold')
    parser.add_argument("-fltnum", "--filter_num", action="store", dest='filter_num', default='256-128',
                    help="filter number")
    parser.add_argument("-fltsize", "--kernel_size", action="store", dest='kernel_size', default='3-3',
                    help="filter size")     
    parser.add_argument("-cnndrop", "--cnndrop_out", action="store", dest='cnndrop_out', default=0.3, type=float,
                    help="cnn drop out")
    parser.add_argument("-rnnsize", "--rnn_size", action="store", dest='rnn_size', default=128, type=int,
                    help="rnn size")                
    parser.add_argument("-fc", "--fc_size", action="store", dest='fc_size', default=32, type=int,
                    help="fully connected size")
    return parser.parse_args()


class UnlabelTestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = row.drop(['config', 'position', 'motif', 'read_index']).values.astype(float)
        return torch.tensor(features.reshape(-1, 9), dtype=torch.float32)


class ConvNet_BiLSTM(nn.Module):
    def __init__(self, output_dim, args, wordvec_len):
        super().__init__()
        chan1, chan2 = map(int, args.filter_num.split('-'))
        ker1, ker2 = map(int, args.kernel_size.split('-'))

        self.conv_layers = nn.Sequential(
            nn.Conv1d(wordvec_len, chan1, kernel_size=ker1, padding=ker1 // 2),
            nn.BatchNorm1d(chan1),
            nn.ReLU(),
            nn.Dropout(args.cnndrop_out),
            nn.Conv1d(chan1, chan2, kernel_size=ker2, padding=ker2 // 2),
            nn.BatchNorm1d(chan2),
            nn.ReLU(),
            nn.Dropout(args.cnndrop_out),
        )

        self.lstm = nn.LSTM(
            input_size=chan2,
            hidden_size=args.rnn_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )

        fc_layers = [
            nn.Linear(args.rnn_size * 2, args.fc_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(args.fc_size, output_dim),
            nn.Sigmoid()
        ] if args.fc_size > 0 else [nn.Linear(args.rnn_size * 2, output_dim)]

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x, args):
        x = self.conv_layers(x.float())
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :]).squeeze()


def get_read_prob(file_path, model, args):
    base_cols = ['config', 'position', 'motif', 'read_index']
    one_hot_cols = [f'one_hot{i}' for i in range(1, 37)]
    stat_cols = ['mean', 'stdv', 'length', 'max', 'min', 'medium']
    max_amp_cols = [f'max_amp{i}' for i in range(1, 10)]
    delta_cols = [f'{feat}{offset:+d}' for feat in ['mean', 'std', 'length', 'max', 'min', 'medium']
                  for offset in [-4, -3, -2, -1, 1, 2, 3, 4]]
    all_columns = base_cols + one_hot_cols + stat_cols + max_amp_cols + delta_cols
    
    data = pd.read_csv(file_path, sep='\t', header=None, names=all_columns, dtype={'config': str})

    reorder_columns = (
        base_cols + one_hot_cols +
        [f'mean{offset:+d}' for offset in [-4, -3, -2, -1]] + ['mean'] + [f'mean{offset:+d}' for offset in [1, 2, 3, 4]] +
        [f'std{offset:+d}' for offset in [-4, -3, -2, -1]] + ['stdv'] + [f'std{offset:+d}' for offset in [1, 2, 3, 4]] +
        [f'length{offset:+d}' for offset in [-4, -3, -2, -1]] + ['length'] + [f'length{offset:+d}' for offset in [1, 2, 3, 4]] +
        [f'max{offset:+d}' for offset in [-4, -3, -2, -1]] + ['max'] + [f'max{offset:+d}' for offset in [1, 2, 3, 4]] +
        [f'min{offset:+d}' for offset in [-4, -3, -2, -1]] + ['min'] + [f'min{offset:+d}' for offset in [1, 2, 3, 4]] +
        [f'medium{offset:+d}' for offset in [-4, -3, -2, -1]] + ['medium'] + [f'medium{offset:+d}' for offset in [1, 2, 3, 4]] +
        max_amp_cols
    )
    data = data[reorder_columns]

    meta = data[base_cols].copy()

    dataset = UnlabelTestDataset(data)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            preds = model(batch, args)
            all_preds.extend(preds.cpu().numpy().flatten())

    meta = meta.assign(probability=np.round(all_preds, 4))
    return meta


def get_site_ratio(read_results, args):
    read_results.loc[:, 'coverage'] = read_results.groupby(['config', 'position', 'motif'])['config'].transform('size')
    read_results = read_results[read_results['coverage'] >= args.min_coverage].copy()
    read_results.loc[:, 'mod_ratio'] = read_results.groupby(['config', 'position', 'motif'])['probability'].transform(
        lambda x: (x >= args.mod_prob_thresh).mean()
    )
    read_results['mod_ratio'] = read_results['mod_ratio'].round(4)
    site_result = read_results[['config', 'position', 'motif', 'mod_ratio', 'coverage']].drop_duplicates()
    return site_result

def process_file(file_path, checkpoint, args, device):
    model = ConvNet_BiLSTM(output_dim=1, args=args, wordvec_len=11).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return get_read_prob(file_path, model, args)


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint = torch.load(os.path.join(args.model_dir, 'checkpoint.pth.tar'), weights_only=False)
    model = ConvNet_BiLSTM(output_dim=1, args=args, wordvec_len=11)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    input_files = sorted([
        os.path.join(args.data_pre, f)
        for f in os.listdir(args.data_pre)
        if f.endswith('.csv') or f.endswith('.txt')
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    read_results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_file)(file_path, checkpoint, args, device)
        for file_path in input_files
    )

    all_read_results = pd.concat(read_results, ignore_index=True)
    all_read_results.to_csv(os.path.join(args.out_dir, 'read_level_result.csv'), sep='\t', index=False)
    site_result = get_site_ratio(all_read_results, args)
    site_result.to_csv(os.path.join(args.out_dir, 'site_level_result.csv'), sep='\t', index=False)

    print(f"✅ Prediction done. Results saved to {args.out_dir}")


if __name__ == '__main__':
    main()
