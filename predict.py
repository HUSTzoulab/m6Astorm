import os
import sys
import argparse
import warnings
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pre', type=str, required=True, help='Path to preprocessed data')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save results')
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
        features = row.drop(['ID', 'Pos', 'motif', 'read_index']).values.astype(float)
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


def get_site_ratio(model, dataloader, meta, args):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            preds = model(batch, args)
            predictions.extend(preds.cpu().numpy().tolist())

    meta = meta.copy()
    meta['pred'] = np.round(predictions, 4)
    read_result = meta.copy()
    meta['coverage'] = meta.groupby(['ID', 'Pos', 'motif'])['ID'].transform('size')
    meta = meta[meta['coverage'] >= args.min_coverage]
    meta['pred_ratio'] = meta.groupby(['ID', 'Pos', 'motif'])['pred'].transform(
        lambda x: (x >= args.mod_prob_thresh).mean()
    )
    meta['pred_ratio'] = meta['pred_ratio'].round(4)

    site_result = meta[['ID', 'Pos', 'motif', 'pred_ratio', 'coverage']].drop_duplicates()
    return read_result, site_result


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint = torch.load(os.path.join(args.model_dir, 'checkpoint.pth.tar'), weights_only=False)

    data = pd.read_csv(args.data_pre, sep='\t', dtype={'ID': str})
    meta = data[['ID', 'Pos', 'motif', 'read_index']]

    dataset = UnlabelTestDataset(data)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    model = ConvNet_BiLSTM(output_dim=1, args=args, wordvec_len=11)
    model.load_state_dict(checkpoint['state_dict'])

    read_result, site_result = get_site_ratio(model, dataloader, meta, args)

    read_result.to_csv(os.path.join(args.out_dir, 'read_level_result.txt'), sep='\t', index=False)
    site_result.to_csv(os.path.join(args.out_dir, 'site_level_result.txt'), sep='\t', index=False)


if __name__ == '__main__':
    main()
