import torch.utils.data as data
import torch
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, datapath, labelspath=None):
        self.data = torch.from_numpy(np.load(datapath)).float()
        self.labels = None if labelspath is None else torch.from_numpy(np.load(labelspath)).float()

    def __getitem__(self, index):
        data = self.data[index]
        if self.labels is not None:
            labels = self.labels[index]
            return data, labels
        else:
            return data


    def __len__(self):
        return len(self.data)