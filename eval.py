import os
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from data_preprocess.process_tabular import Data_Loader
from network.attention import Attention
import yaml

class DatasetBuilder(Dataset):
    def __init__(self, data, subset_size, overlap, label=None, norm='minmax'):
        self.data = data
        self.label = label
        self.subset_size = subset_size
        self.overlap = overlap
        self.norm = norm

    def normalization(self, data):
        if self.norm == 'minmax':
            if (torch.max(data) - torch.min(data)) == 0:
                eps = 1e-8
                data = (data - torch.min(data) + eps) / (torch.max(data) - torch.min(data) + eps)
            else:
                data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
            return data

    def __len__(self):
        return (self.data.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        data = self.normalization(data)
        subset_size = self.subset_size
        overlap = self.overlap
        stride = subset_size - overlap
        data = data.unfold(0, subset_size, stride)
        if self.label == None:
            return {'data': data, 'index': idx}
        else:
            return {'data': data, 'label': self.label[idx], 'index': idx}

def main(args):
    tabular_loader = Data_Loader()
    train_data, test_data, test_labels = tabular_loader.get_dataset(args.dataset)
    test_data = torch.as_tensor(test_data, dtype=torch.float)
    test_dataset = DatasetBuilder(test_data, args.subset_size, args.overlap, label=test_labels, norm=args.normalization)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = Attention(dim=args.hidden_dim, att_dim=args.subset_size)
    model.load_state_dict(torch.load(f'{args.checkpoints_path}{args.dataset}.pth'))
    model.to(device)

    model.eval()
    label_score = []
    for i, sample in enumerate(test_loader):
        data = sample['data'].to(device)
        target = sample['label'].to(device)
        with torch.no_grad():
            anomaly_score = model(data, labels=target)
            label_score += list(zip(target.cpu().data.numpy().tolist(), anomaly_score.cpu().data.numpy().tolist()))
    labels, scores = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    n_samples = labels.shape[0]
    n_anomaly = np.sum(labels == 1)
    thresh = np.percentile(scores, 100 * (n_anomaly / n_samples))
    y_pred = np.where(scores <= thresh, 1, 0)
    prec, recall, f1, _ = precision_recall_fscore_support(labels, y_pred, average="binary")
    auc_roc = roc_auc_score(labels, -scores)
    print(f1, auc_roc)

    return f1, auc_roc


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    parser = argparse.ArgumentParser(description='Tabular Training')
    parser.add_argument('--dataset', type=str, default='thyroid')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--subset_size', type=int, default=1)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--normalization', type=str, default='minmax')
    parser.add_argument('--checkpoints_path', type=str, default='checkpoints/')
    parser.add_argument('--config_name', type=str, default='tabular_config.yaml')
    args = parser.parse_args()

    with open(f'config/{args.config_name}', 'r') as f:
        config_file = yaml.safe_load(f)

    config = config_file[args.dataset]
    args.batch_size = config['batch_size']
    args.subset_size = config['subset_size']
    args.overlap = config['overlap']
    args.hidden_dim = config['hidden_dim']
    args.normalization = config['normalization']

    print(args)
    main(args)