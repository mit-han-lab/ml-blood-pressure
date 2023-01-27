from typing import List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torchpack.utils.config import configs

import numpy as np

__all__ = ['Lstm']


class Lstm(nn.Module):
    def __init__(self, in_ch, out_ch, bidirectional, layer_num, dropout=0.1):
        super().__init__()

        self.input_layer = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1)

        self.lstm = nn.LSTM(
            input_size=out_ch,
            hidden_size=out_ch,
            num_layers=layer_num,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        
        self.regress = nn.Linear(2*out_ch if bidirectional else out_ch, 1)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x):
        if 'v' in x.keys():
            seq_len = x['v'].shape[1]
        elif 'bp' in x.keys():
            seq_len = x['bp'].shape[1]
        elif 'velocity_complete_avg_padded' in x.keys():
            seq_len = x['velocity_complete_avg_padded'].shape[1]
        else:
            seq_len = x['bp'].shape[1]
            # raise ValueError

        feats = []

        # vector features
        for feat in ['area', 'v', 'bp', 'shape', 'flow', 'diameter_complete_avg_beats', 'velocity_complete_avg_beats',
                     'bp_shape_complete_avg_beats', 'area_complete_avg_beats']:
            if feat in configs.model.feats:
                feats.append(x[feat])

        # scalar features
        for feat in ['age', 'weight', 'height', 'gender', 'pwv', 'pp', 'comp', 'z0', 'deltat', 'heartrate',
                     'bp_shape_complete_min', 'bp_shape_complete_mean', 'bp_shape_complete_max',
                     'velocity_complete_min', 'velocity_complete_mean', 'velocity_complete_max']:
            if feat in configs.model.feats:
                feat_vect = torch.unsqueeze(x[feat], dim=-1).repeat([1, seq_len])
                feats.append(feat_vect)

        x = torch.stack(feats, dim=-1)

        x = x.permute(0, 2, 1)
        x = self.input_layer(x)  # N, C, L
        x = x.permute(2, 0, 1)
        x = self.dropout1(x)

        x = self.lstm(x)[0]  # output: L, N, C
        x = x.permute(1, 2, 0)  # N, C, L
        x = self.dropout2(x)

        x = F.avg_pool1d(x, kernel_size=x.shape[-1]).squeeze()
        x = self.regress(x)

        return x.squeeze(dim=-1)
