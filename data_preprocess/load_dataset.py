import torch
from torch.utils.data import Dataset

class DatasetBuilder(Dataset):
    def __init__(self, data, patch_size, overlap,  norm, label=None):
        self.data = data
        self.label = label
        self.patch_size = patch_size
        self.overlap = overlap
        self.norm = norm

    def normalization(self, data):
        if self.norm == 'minmax':
            data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
            return data
        elif self.norm == 'std':
            data = (data - torch.mean(data)) / torch.std(data)
            return data
        else:
            return data

    def __len__(self):
        return (self.data.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        data = self.normalization(data)

        subset_size = self.patch_size
        overlap = self.overlap
        stride = subset_size - overlap
        data = data.unfold(0, subset_size, stride)

        if self.label == None:
            return {'data': data, 'index': idx}
        else:
            return {'data': data, 'label': self.label[idx], 'index': idx}