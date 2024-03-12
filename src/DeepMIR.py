import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class SpatialPyramidPooling(nn.Module):
    def __init__(self, levels):
        super(SpatialPyramidPooling, self).__init__()
        self.levels = levels

    def forward(self, x):
        N, C, H, W = x.shape
        bins = []
        for l in range(self.levels):
            kh = int(np.ceil(H / (l + 1)))
            kw = int(np.ceil(W / (l + 1)))
            sh = int(np.floor(H / (l + 1)))
            sw = int(np.floor(W / (l + 1)))
            pool = nn.MaxPool2d(kernel_size=(kh, kw), stride=(sh, sw))
            bins.append(pool(x))
        out = torch.cat([bin.view(N, -1) for bin in bins], dim=1)
        return out

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.Conv1dA = nn.Conv1d(1, 32, 7, 1, 3)
        self.BN1dA = nn.BatchNorm1d(32)
        self.Pool1dA = nn.MaxPool1d(3)
        self.Conv1dB = nn.Conv1d(1, 32, 7, 1, 3)
        self.BN1dB = nn.BatchNorm1d(32)
        self.Pool1dB = nn.MaxPool1d(3)
        self.Conv2d = nn.Conv2d(1, 32, 7, 1, 3)
        self.BN2d = nn.BatchNorm2d(32)
        self.Pool2d = SpatialPyramidPooling(4)
        self.Flatten1 = nn.Linear(960, 1024)
        self.drop = nn.Dropout(0.5)
        self.Flatten2 = nn.Linear(1024, 1)

    def forward(self, x):
        inputA = x[:, 0, None, :]
        inputB = x[:, 1, None, :]
        Conv1dA = self.Conv1dA(inputA)
        BN1dA = self.BN1dA(Conv1dA)
        Act1dA = F.relu(BN1dA)
        Pool1dA = self.Pool1dA(Act1dA)
        Conv1dB = self.Conv1dB(inputB)
        BN1dB = self.BN1dA(Conv1dB)
        Act1dB = F.relu(BN1dB)
        Pool1dB = self.Pool1dA(Act1dB)
        con = torch.cat([Pool1dA, Pool1dB], dim=1)
        con = torch.unsqueeze(con, 1)
        Conv2d = self.Conv2d(con)
        BN2d = self.BN2d(Conv2d)
        Act2d1 = F.relu(BN2d)
        Pool2d = self.Pool2d(Act2d1)
        Flatten1 = self.Flatten1(Pool2d)
        Act2d2 = F.relu(Flatten1)
        drop = self.drop(Act2d2)
        Flatten2 = self.Flatten2(drop)
        outputs = torch.sigmoid(Flatten2)
        return outputs