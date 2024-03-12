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
        self.Conv1d = nn.Conv1d(1, 32, 7, 1, 3)
        self.BN1d = nn.BatchNorm1d(32)
        self.Act1d = nn.ReLU()
        self.Pool1d = nn.MaxPool1d(3)


        #ASPP
        self.Conv2d1 = nn.Conv2d(1, 32, 1, 1)
        self.BN2d1 = nn.BatchNorm2d(32)
        self.Act2d1 = nn.ReLU()
        self.Conv2d2 = nn.Conv2d(32, 32, 3, 1, 2, 2)
        self.BN2d2 = nn.BatchNorm2d(32)
        self.Act2d2 = nn.ReLU()
        self.Conv2d3 = nn.Conv2d(32, 32, 3, 1, 4, 4)
        self.BN2d3 = nn.BatchNorm2d(32)
        self.Act2d3 = nn.ReLU()
        self.Conv2d4 = nn.Conv2d(32, 32, 1, 1)
        self.BN2d4 = nn.BatchNorm2d(32)
        self.Act2d4 = nn.ReLU()

        self.Pool2d = SpatialPyramidPooling(4)
        self.Flatten1 = nn.Linear(960, 1024)
        self.Act2d5 = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.Flatten2 = nn.Linear(1024, 1)
        self.Act2d6 = nn.Sigmoid()

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
        Conv1dA = self.Conv1d(inputA)
        BN1dA = self.BN1d(Conv1dA)
        Act1dA = self.Act1d(BN1dA)
        Pool1dA = self.Pool1d(Act1dA)
        Conv1dB = self.Conv1d(inputB)
        BN1dB = self.BN1d(Conv1dB)
        Act1dB = self.Act1d(BN1dB)
        Pool1dB = self.Pool1d(Act1dB)
        con = torch.cat([Pool1dA, Pool1dB], dim=1)
        con = torch.unsqueeze(con, 1)

        Conv2d1 = self.Conv2d1(con)
        BN2d1 = self.BN2d1(Conv2d1)
        Act2d1 = self.Act2d1(BN2d1)
        Conv2d2 = self.Conv2d2(Act2d1)
        BN2d2 = self.BN2d2(Conv2d2)
        Act2d2 = self.Act2d2(BN2d2)
        Conv2d3 = self.Conv2d3(Act2d2)
        BN2d3 = self.BN2d3(Conv2d3)
        Act2d3 = self.Act2d3(BN2d3)
        Conv2d4 = self.Conv2d4(Act2d3)
        BN2d4 = self.BN2d4(Conv2d4)
        Act2d4 = self.Act2d4(BN2d4)

        Pool2d = self.Pool2d(Act2d4)
        Flatten1 = self.Flatten1(Pool2d)
        Act2d5 = self.Act2d5(Flatten1)
        drop = self.drop(Act2d5)
        Flatten2 = self.Flatten2(drop)
        outputs = self.Act2d6(Flatten2)
        return outputs


