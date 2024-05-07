import numpy as np
import torch
import torch.nn as nn

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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Conv1dA = nn.Conv1d(1, 32, 7, 1, 3)
        self.BN1dA = nn.BatchNorm1d(32)
        self.Act1dA = nn.ReLU()
        self.Pool1dA = nn.MaxPool1d(3)
        self.Conv1dB = nn.Conv1d(1, 32, 7, 1, 3)
        self.BN1dB = nn.BatchNorm1d(32)
        self.Act1dB = nn.ReLU()
        self.Pool1dB = nn.MaxPool1d(3)
        self.Conv2d = nn.Conv2d(1, 32, 7, 1, 3)
        self.BN2d = nn.BatchNorm2d(32)
        self.Act2d1 = nn.ReLU()
        self.Pool2d = SpatialPyramidPooling(4)
        self.Flatten1 = nn.Linear(960, 1024)
        self.Act2d2 = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.Flatten2 = nn.Linear(1024, 1)
        self.Act2d3 = nn.Sigmoid()
        nn.init.kaiming_normal_(self.Flatten1.weight, mode='fan_in', nonlinearity='relu')
        # Apply Xavier initialization
        nn.init.xavier_normal_(self.Flatten2.weight)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    def forward(self, x):
        inputA = x[:, 0, None, :]
        inputB = x[:, 1, None, :]
        Conv1dA = self.Conv1dA(inputA)
        BN1dA = self.BN1dA(Conv1dA)
        Act1dA = self.Act1dA(BN1dA)
        Pool1dA = self.Pool1dA(Act1dA)
        Conv1dB = self.Conv1dB(inputB)
        BN1dB = self.BN1dB(Conv1dB)
        Act1dB = self.Act1dB(BN1dB)
        Pool1dB = self.Pool1dB(Act1dB)
        con = torch.cat([Pool1dA, Pool1dB], dim=1)
        con = torch.unsqueeze(con, 1)
        Conv2d = self.Conv2d(con)
        BN2d = self.BN2d(Conv2d)
        Act2d1 = self.Act2d1(BN2d)
        Pool2d = self.Pool2d(Act2d1)
        Flatten1 = self.Flatten1(Pool2d)
        Act2d2 = self.Act2d2(Flatten1)
        drop = self.drop(Act2d2)
        Flatten2 = self.Flatten2(drop)
        outputs = self.Act2d3(Flatten2)
        return outputs


