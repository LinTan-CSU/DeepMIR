import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class SpectralPositionalEncoding(nn.Module):
    """
    * Spectral Positional Encoding module for sequence data
    *
    * Attributes
    * ----------
    * d_model : int
    *     The dimension of the model (feature size).
    * dropout : float
    *     Dropout rate applied after adding positional encoding.
    * max_len : int
    *     Maximum sequence length supported.
    *
    * Methods
    * -------
    * forward(x)
    *     Add positional encoding to the input tensor.
    *
    * Parameters
    * ----------
    * x : torch.Tensor
    *     Input tensor of shape (batch_size, sequence_length, d_model)
    *
    * Returns
    * -------
    * torch.Tensor
    *     Output tensor with positional encoding added, same shape as input.
    """
    def __init__(self, d_model, dropout=0.1, max_len=3000):
        super(SpectralPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Model(nn.Module):
    """
    * Construct the DeepMIR model
    *
    * Attributes
    * ----------
    * d_model : int
    *     The dimension of the model (feature size for transformer and last conv layer).
    * nhead : int
    *     Number of attention heads in the transformer encoder.
    * num_layers : int
    *     Number of transformer encoder layers.
    * trans_dropout : float
    *     Dropout rate for the transformer encoder.
    * mlp_dropout : float
    *     Dropout rate for the MLP classifier.
    *
    * Methods
    * -------
    * forward(inputs)
    *     Forward pass of the model.
    *
    * Parameters
    * ----------
    * inputs : torch.Tensor
    *     Input tensor of shape (batch_size, 2, sequence_length)
    *
    * Returns
    * -------
    * output : torch.Tensor
    *     Output tensor of shape (batch_size, 1)
    """
    def __init__(self, d_model=128, nhead=8, num_layers=2, trans_dropout=0.19, mlp_dropout=0.28):
        super(Model, self).__init__()
        self.Conv1 = nn.Conv1d(2, 32, 5, 1, 2)
        self.BN1 = nn.BatchNorm1d(32)
        self.Conv2 = nn.Conv1d(32, 64, 5, 1, 2)
        self.BN2 = nn.BatchNorm1d(64)
        self.Conv3 = nn.Conv1d(64, d_model, 3, 1, 1)
        self.BN3 = nn.BatchNorm1d(d_model)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        self.spec_pos = SpectralPositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=trans_dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, inputs):
        ref = inputs[:,0,None,:]
        aug = inputs[:,1,None,:]
        x = torch.cat([ref, aug], dim=1)  # [B, 2, L]
        x = self.act(self.BN1(self.Conv1(x)))
        x = self.pool(x)
        x = self.act(self.BN2(self.Conv2(x)))
        x = self.pool(x)
        x = self.act(self.BN3(self.Conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # [B, seq, feature]
        x = self.spec_pos(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        output = self.mlp(x)
        return output
