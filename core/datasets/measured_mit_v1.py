import os
import random
import sys
import pdb
import scipy
import scipy.signal
from typing import List, Optional, Tuple, Callable

import numpy as np
import torch
from PIL import Image
from torchpack.datasets.dataset import Dataset
from torchpack.utils import imp, io
from torchvision.transforms import functional as F
from tools import read_scalar, read_vector


__all__ = ['MeasuredMITV1Dataset', 'MeasuredMITV1']


class MeasuredMITV1Dataset:
    def __init__(self,
                 root: str,
                 split: str,
                 split_ratio: List[float],
                 custom_split_indices: List[List[int]],  # precedent over split ratio
                 beats_per_subject: int,
                 part: int,
                 ensembled: bool,
                 mean: dict = None,
                 std: dict = None,
                 subject_name: str = None,
                 target: str = 'map',
                 ):
        super().__init__()
        self.target = target
        self.root = root
        self.split = split
        self.split_ratio = split_ratio
        self.custom_split_indices = custom_split_indices
        print("split", split, split_ratio)
        print("custom split indices", split, custom_split_indices)
        self.raw = {}
        self.orig = {}
        self.scalar_feat_name = ['map',
                                 'map_alt',
                                 # self.target,
                                 'name',
                                 'id',
                                 'age',
                                 'dbp',
                                 'gender',
                                 'height',
                                 'weight',
                                 'heartrate',
                                 'map_complete_avg_beats',
                                 'bp_shape_complete_min',
                                 'bp_shape_complete_mean',
                                 'bp_shape_complete_max',
                                 'velocity_complete_min',
                                 'velocity_complete_mean',
                                 'velocity_complete_max',
                                 ]
        self.vector_feat_name = ['shape', 'v', 'area', 'velocity_complete_avg_beats', 'diameter_complete_avg_beats',
                                 'bp_shape_complete_avg_beats', 'area_complete_avg_beats']
        # contains data that is different per beat
        self.multi_beats_name = ['velocity_complete_avg_beats', 'diameter_complete_avg_beats',
                                 'bp_shape_complete_avg_beats', 'area_complete_avg_beats', 'map_complete_avg_beats']
        self.mean = {}
        self.std = {}
        self.peak_idx = None
        self.subject_name = subject_name
        self.part = part
        self.ensembled = ensembled

        self.beats_per_subject = beats_per_subject
        self.beats_per_subject_indiv = []
        self.numericals = ['map',
                           'map_alt',
                           # self.target,
                           'shape',
                           'age',
                           'dbp',
                           'gender',
                           'height',
                           'weight',
                           'heartrate',
                           'v',
                           'area',
                           'velocity_complete_avg_beats',
                           'diameter_complete_avg_beats',
                           'bp_shape_complete_avg_beats', 
                           'area_complete_avg_beats',
                           'map_complete_avg_beats',
                           'bp_shape_complete_min',
                           'bp_shape_complete_mean',
                           'bp_shape_complete_max',
                           'velocity_complete_min',
                           'velocity_complete_mean',
                           'velocity_complete_max',
                           ]

        self._load()
        if not self.ensembled:
            self._reshape_beats()
        self._get_mean_std()
        # overwrite mean and std options if given
        if mean is not None:
            print("using given mean")
            for k, v in mean.items():
                self.mean[k] = v
        if std is not None:
            print("using given std")
            for k, v in std.items():
                self.std[k] = v
        self._normalize()
        self._split()

        self.instance_num = len(self.raw[list(self.raw.keys())[0]])

    def _load(self):
        for feat_name in self.scalar_feat_name + self.vector_feat_name:
            # if in ensembled mode, don't data for load individual beats
            if self.ensembled and feat_name in self.multi_beats_name:
                continue
            file_name = os.path.join(self.root, f'measured_mit_v1_part{self.part}_{feat_name}_all.npy')
            self.raw[feat_name] = np.load(file_name, allow_pickle=True)

    def _split(self):
        if self.custom_split_indices is None:  # use split ratio
            print("using split ratio")
            instance_num =len(self.raw[list(self.raw.keys())[0]])
            split_train = self.split_ratio[0]
            split_valid = self.split_ratio[0] + self.split_ratio[1]
            if self.split == 'train':
                for feat_name in self.raw.keys():
                    self.raw[feat_name] = self.raw[feat_name][
                                          :int(split_train * instance_num)]
            elif self.split == 'valid':
                for feat_name in self.raw.keys():
                    self.raw[feat_name] = self.raw[feat_name][
                                          int(split_train * instance_num):
                                          int(split_valid * instance_num)]
            elif self.split == 'test':
                for feat_name in self.raw.keys():
                    self.raw[feat_name] = self.raw[feat_name][
                                          int(split_valid * instance_num):]
            else:
                raise ValueError(self.split)
        else:  # use custom split indices
            print("using custom split indices")
            if self.split == 'train':
                indices = self.custom_split_indices[0]
            elif self.split == 'valid':
                indices = self.custom_split_indices[1]
            elif self.split == 'test':
                indices = self.custom_split_indices[2]
            else:
                raise ValueError(self.split)

            for feat_name in self.raw.keys():
                self.raw[feat_name] = self.raw[feat_name][indices]

    def _get_mean_std(self):
        for feat_name in self.raw.keys():
            if feat_name in self.numericals:
                print(feat_name, self.raw[feat_name].shape)
                self.mean[feat_name] = np.mean(self.raw[feat_name])
                self.std[feat_name] = np.std(self.raw[feat_name])

    def _normalize(self):
        for feat_name in self.raw.keys():
            if feat_name in self.numericals:
                self.raw[feat_name] = (self.raw[feat_name] -
                                       self.mean[feat_name]) / \
                                      self.std[feat_name]

    def _reshape_beats(self):
        """
        reshape data such that each entry is an individual beat, not a whole subject
        """
        beats_per_subject = []
        for i in range(len(self.raw[next(iter(self.raw.keys()))])):
            # get num beats for given subject
            num_beats = len(self.raw[next(iter(self.multi_beats_name))][i])
            beats_per_subject.append(num_beats)
        self.beats_per_subject_indiv = beats_per_subject

        for feat_name in self.raw.keys():
            all_beats = []
            if feat_name in self.multi_beats_name:
                for i in range(len(self.raw[feat_name])):
                    for beat in self.raw[feat_name][i]:
                        all_beats.append(beat)
                self.raw[feat_name] = np.array(all_beats)
            else:
                for i in range(len(self.raw[feat_name])):
                    for _ in range(beats_per_subject[i]):
                        all_beats.append(self.raw[feat_name][i])
                self.raw[feat_name] = np.array(all_beats)

    def __getitem__(self, index: int):
        data_this = {}

        subject_id = index

        for feat_name in self.raw.keys():
            if feat_name in self.numericals:
                data_this[feat_name] = self.raw[feat_name][subject_id].astype(np.float32)
            else:
                data_this[feat_name] = self.raw[feat_name][subject_id]

        return data_this

    def __len__(self) -> int:
        return self.instance_num


class MeasuredMITV1(Dataset):
    def __init__(self,
                 root: str,
                 split_ratio: List[float],
                 beats_per_subject,
                 part: int,
                 ensembled: bool,
                 custom_split_indices: List[List[int]]=None,
                 mean=None,
                 std=None,
                 subject_name=None,
                 target='map',
                 ):
        self.root = root

        super().__init__({
            split: MeasuredMITV1Dataset(root=root,
                             split=split,
                             split_ratio=split_ratio,
                             part=part,
                             ensembled=ensembled,
                             beats_per_subject=beats_per_subject,
                             custom_split_indices=custom_split_indices,
                             mean=mean,
                             std=std,
                             subject_name=subject_name,
                             target=target,
                            )
            for split in ['train', 'valid', 'test']

        })

        self.beats_per_subject_indiv = self['train'].beats_per_subject_indiv


if __name__ == '__main__':
    import matplotlib
    # matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    pdb.set_trace()
    pd = MeasuredMITV1Dataset(root='./data/measured_mit_v1/npy/',
                   split='test',
                   split_ratio=[0.7, 0.1, 0.2],
                   beats_per_subject=1,
                   part=1,
                   )

    # plt.plot(pd.raw['a'][0])
    # plt.show()
    print(pd[0])
    print('Finish')
