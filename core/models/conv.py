from typing import List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torchpack.utils.config import configs

import numpy as np

__all__ = ['Conv']


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, layer_num, dropout=0.0):
        super().__init__()

        self.input_layer = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1)

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(nn.Conv1d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ))
        self.dropout_layers = nn.ModuleList()
        for i in range(layer_num):
            self.dropout_layers.append(nn.Dropout(p=dropout))
        self.regress = nn.Linear(out_ch, 1)

    def forward(self, x):
        seq_len = x['v'].shape[1]

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

        x = x.permute(0, 2, 1)  # N, C, L
        x = self.input_layer(x)

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))
            x = self.dropout_layers[i](x)

        x = F.avg_pool1d(x, kernel_size=x.shape[-1]).squeeze()
        x = self.regress(x)

        return x.squeeze(dim=-1)
