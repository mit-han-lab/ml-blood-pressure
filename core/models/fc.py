from typing import List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torchpack.utils.config import configs

__all__ = ['FC']

class FC(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 layer_num: int,
                 dropout: float,
                 ):
        super().__init__()
        self.input_layer = nn.Linear(in_ch, out_ch)

        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(nn.Linear(out_ch, out_ch))
            self.dropout_layers.append(nn.Dropout(p=dropout))

        self.regress = nn.Linear(out_ch, 1)

    def forward(self, x):
        feats = []
        # vector features - use mean
        for feat in ['area', 'v', 'bp', 'shape', 'flow', 'diameter_complete_avg_beats', 'velocity_complete_avg_beats',
                     'bp_shape_complete_avg_beats', 'area_complete_avg_beats']:
            if feat in configs.model.feats:
                feats.append(torch.mean(x[feat], dim=-1))

        # scalar features
        for feat in ['age', 'weight', 'height', 'gender', 'pwv', 'pp', 'comp', 'z0', 'deltat', 'heartrate',
                     'bp_shape_complete_min', 'bp_shape_complete_mean', 'bp_shape_complete_max',
                     'velocity_complete_min', 'velocity_complete_mean', 'velocity_complete_max']:
            if feat in configs.model.feats:
                feats.append(x[feat])

        x = torch.stack(feats, dim=-1)

        x = self.input_layer(x)  # N, C

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))
            x = self.dropout_layers[i](x)

        x = self.regress(x)

        return x.squeeze(dim=-1)


class FC_bp2cvp_single(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 layer_num: int,
                 ):
        super().__init__()
        self.input_layer = nn.Linear(in_ch, out_ch)

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(nn.Linear(out_ch, out_ch))

        self.regress = nn.Linear(out_ch, 1)

    def forward(self, x):
        # x = torch.stack((x['pwv'],
        #                  x['comp'],
        #                  x['z0'],
        #                  x['deltat'],
        #                  x['pp'],
        #                  torch.mean(x['a'], dim=-1),
        #                  torch.mean(x['v'], dim=-1)), dim=-1)
        x = x['bp']

        x = self.input_layer(x)  # N, C

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))

        x = self.regress(x)

        return x.squeeze(dim=-1)


class FC_bp2cvp_common(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 layer_num: int,
                 ):
        super().__init__()
        self.input_layer = nn.Linear(in_ch, out_ch)

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(nn.Linear(out_ch, out_ch))

        self.regress = nn.Linear(out_ch, 1)

    def forward(self, x):
        # x = torch.stack((x['pwv'],
        #                  x['comp'],
        #                  x['z0'],
        #                  x['deltat'],
        #                  x['pp'],
        #                  torch.mean(x['a'], dim=-1),
        #                  torch.mean(x['v'], dim=-1)), dim=-1)
        x = torch.stack((x['age'],
                         x['sex'],
                         x['height'],
                         x['weight'],
                         x['bmi'],
                         x['bp'],
                         ), dim=-1).float()
        # x = x['bp']

        x = self.input_layer(x)  # N, C

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))

        x = self.regress(x)

        return x.squeeze(dim=-1)
