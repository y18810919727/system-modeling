#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch

from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from common import subsample_indexes
from common import onceexp


class WesternDataset(Dataset):
    def __init__(self, df_list, length=1000, step=5, dilation=2):
        """

        Args:
            df_list:
            length:
            step: 数据segment切割窗口的移动步长
            dilation: 浓密机数据采样频率(1 min)过高，dilation表示数据稀释间距
        """
        if not isinstance(df_list, list):
            df_list = [df_list]
        df_split_all = []
        begin_pos_pair = []

        # 每个column对应的数据含义 ['c_in','c_out', 'v_out', 'v_in', 'pressure']
        self.used_columns = ['4', '11', '14', '16', '17']
        self.length = length
        self.dilation = dilation

        for df in df_list:
            df_split_all = df_split_all + self.split_df(df[self.used_columns])
        for i, df in enumerate(df_split_all):
            for j in range(0, df.shape[0] - length * dilation + 1, step):
                begin_pos_pair.append((i, j))
        self.begin_pos_pair = begin_pos_pair
        self.df_split_all = df_split_all
        self.df_split_all = self.normalize(self.df_split_all)

    def normalize(self, df_all_list):
        df_all = df_all_list[0].append(df_all_list[1:], ignore_index=True)
        mean = df_all.mean()
        std = df_all.std()
        return [(df - mean) / std for df in df_all_list]

    def split_df(self, df):
        """
        将存在空值的位置split开
        Args:
            df:
        Returns: list -> [df1,df2,...]
        """
        df_list = []
        split_indexes = list(
            df[df.isnull().T.any()].index
        )
        split_indexes = [-1] + split_indexes + [df.shape[0]]
        for i in range(len(split_indexes) - 1):
            if split_indexes[i + 1] - split_indexes[i] - 1 < self.length:
                continue

            new_df = df.iloc[split_indexes[i] + 1:split_indexes[i + 1]]
            assert new_df.isnull().sum().sum() == 0
            df_list.append(new_df)
        return df_list

    def __len__(self):
        return len(self.begin_pos_pair)

    def __getitem__(self, item):
        df_index, pos = self.begin_pos_pair[item]
        data_array = np.array(self.df_split_all[df_index].iloc[pos:pos + self.length * self.dilation], dtype=np.float32)
        data_array = data_array[np.arange(self.length) * self.dilation]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        c_in, c_out, v_out, v_in, pressure = [np.squeeze(x, axis=1) for x in np.hsplit(data_array, 5)]

        v_in = v_in * 0.05
        v_out = v_out * 0.05

        external_input = np.stack(
            [
                c_in * c_in * c_in * v_in - c_out * c_out * c_out * v_out,
                c_in * c_in * v_in - c_out * c_out * v_out,
                c_in * v_in - c_out * v_out,
                v_in - v_out,
                v_in,
                v_out,
                c_in,
                c_out
            ],
            axis=1)
        observation = pressure
        return external_input, np.expand_dims(observation, axis=1)


class WesternDataset_1_4(WesternDataset):

    def __len__(self):
        return len(self.begin_pos_pair)

    def __getitem__(self, item):
        df_index, pos = self.begin_pos_pair[item]
        data_array = np.array(self.df_split_all[df_index].iloc[pos:pos + self.length * self.dilation], dtype=np.float32)
        data_array = data_array[np.arange(self.length) * self.dilation]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        c_in, c_out, v_out, v_in, pressure = [np.squeeze(x, axis=1) for x in np.hsplit(data_array, 5)]

        external_input = v_out

        observation = np.stack(
            [
                c_out,
                c_in,
                v_in,
                pressure
            ],
            axis=1)

        return np.expand_dims(external_input, axis=1), observation


class CstrDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        df_split_all = []
        begin_pos = []

        # 每个column对应的数据含义 ['in','out1', 'out2']
        self.df = df
        self.used_columns = ['0', '1', '2']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        data_in = np.array(data_df['0'], dtype=np.float32)
        data_out = np.array(data_df[['1', '2']], dtype=np.float32)

        # return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)
        return np.expand_dims(data_in, axis=1), data_out


class NLDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        df_split_all = []
        begin_pos = []

        # 每个column对应的数据含义 ['input','output']
        self.df = df
        self.used_columns = ['u', 'y']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        data_in = np.array(data_df['u'], dtype=np.float32)
        data_out = np.array(data_df['y'], dtype=np.float32)

        # return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)
        return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)


class IBDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        df_split_all = []
        begin_pos = []
        self.mean = 0.0
        self.std = 0.0
        # 每个column对应的数据含义 ['delta_v', 'delta_g', 'delta_h','f','c', 'reward']
        self.df = df
        self.used_columns = ['delta_v', 'delta_g', 'delta_h', 'v', 'g', 'h', 'f', 'c', 'reward']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        self.mean = df.mean()
        self.std = df.std()

        return (df - self.mean) / self.std

    def normalize_record(self):
        mean = self.mean
        std = self.std
        return [mean, std]

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df[['delta_v', 'delta_g', 'delta_h']], dtype=np.float32)
        data_out = np.array(data_df[['v', 'g', 'h', 'f', 'c', 'reward']], dtype=np.float32)
        # return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)
        return data_in, data_out


class WindingDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        df_split_all = []
        begin_pos = []

        # 每个column对应的数据含义 ['in','out1', 'out2']
        self.df = df
        self.used_columns = ['0', '1', '2', '3', '4', '5', '6']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        data_in = np.array(data_df[['0', '1', '2', '3', '4']], dtype=np.float32)
        data_out = np.array(data_df[['5', '6']], dtype=np.float32)

        # return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)
        return data_in, data_out


class WesternConcentrationDataset(Dataset):
    def __init__(self, df_list, length=1000, step=5, dilation=2):
        """

        Args:
            df_list:
            length:
            step: 数据segment切割窗口的移动步长
            dilation: 浓密机数据采样频率(1 min)过高，dilation表示数据稀释间距
        """
        if not isinstance(df_list, list):
            df_list = [df_list]
        df_split_all = []
        begin_pos_pair = []

        # 每个column对应的数据含义 ['c_in','c_out', 'v_out', 'v_in', 'pressure']
        self.used_columns = ['4', '5', '7', '11', '14', '16', '17']
        self.length = length
        self.dilation = dilation

        for df in df_list:
            df_split_all = df_split_all + self.split_df(df[self.used_columns])
        for i, df in enumerate(df_split_all):
            for j in range(0, df.shape[0] - length * dilation + 1, step):
                begin_pos_pair.append((i, j))
        self.begin_pos_pair = begin_pos_pair
        self.df_split_all = df_split_all
        self.df_split_all = self.normalize(self.df_split_all)

    def normalize(self, df_all_list):
        df_all = df_all_list[0].append(df_all_list[1:], ignore_index=True)
        mean = df_all.mean()
        std = df_all.std()
        return [(df - mean) / std for df in df_all_list]

    def split_df(self, df):
        """
        将存在空值的位置split开
        Args:
            df:
        Returns: list -> [df1,df2,...]
        """
        df_list = []
        split_indexes = list(
            df[df.isnull().T.any()].index
        )
        split_indexes = [-1] + split_indexes + [df.shape[0]]
        for i in range(len(split_indexes) - 1):
            if split_indexes[i + 1] - split_indexes[i] - 1 < self.length:
                continue

            new_df = df.iloc[split_indexes[i] + 1:split_indexes[i + 1]]
            assert new_df.isnull().sum().sum() == 0
            df_list.append(new_df)
        return df_list

    def __len__(self):
        return len(self.begin_pos_pair)

    def __getitem__(self, item):
        df_index, pos = self.begin_pos_pair[item]
        # data_array = np.array(self.df_split_all[df_index].iloc[pos:pos+self.length*self.dilation], dtype=np.float32)
        # data_array = data_array[np.arange(self.length) * self.dilation]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]

        # data_in = np.array(data_array[['4','5','7','14','16']], dtype=np.float32)
        # data_out = np.array(data_array[['11', '17']], dtype=np.float32)

        data_df = self.df_split_all[df_index].iloc[pos:pos + self.length * self.dilation]

        def choose_and_dilation(df, length, dilation, indices):
            return np.array(
                df[indices], dtype=np.float32
            )[np.arange(length) * dilation]

        data_in = choose_and_dilation(data_df, self.length, self.dilation, ['4', '5', '7', '14', '16'])
        data_out = choose_and_dilation(data_df, self.length, self.dilation, ['11', '17'])

        # return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)
        return data_in, data_out


class SoutheastThickener(Dataset):
    def __init__(self, data, length=90, step=5, dilation=1, dataset_type=None, ratio=None, io=None, seed=0,
                 smooth_alpha=0.3):
        """

        Args:
            data: data array
            length: history + predicted
            step:  size of moving step
            dilation:
            dataset_type:  train, val, test
            ratio:  default: 0.6, 0.2, 0.2
            io: default 4-1
            seed: default 0
        """

        # TODO: 支持序列长度非90
        assert length == 90
        if not isinstance(seed, int):
            seed = 0

        if dataset_type is None:
            dataset_type = 'train'

        if ratio is None:
            ratio = [0.6, 0.2, 0.2]

        if io is None:
            io = '4-1'

        if io == '4-1':
            # 进料浓度、出料浓度、进料流量、出料流量 -> 泥层压力
            self.io = [[0, 1, 2, 3], [4]]
        elif io == '3-2':
            # 进料浓度、进料流量、出料流量 -> 出料浓度 、泥层压力
            self.io = [[0, 2, 3], [1, 4]]
        else:
            raise NotImplementedError()

        data = np.array(data, dtype=np.float32)
        self.smooth_alpha = smooth_alpha

        for _ in self.io[1]:
            # data.shape (N, 90, 5)
            data[:, :, int(_)] = onceexp(data[:, :, int(_)].transpose(), self.smooth_alpha).transpose()

        data, self.mean, self.std = self.normalize(data)

        data = data[::step]
        L = data.shape[0]

        train_size, val_size = int(L*ratio[0]), int(L*ratio[1])
        test_size = L - train_size - val_size

        d1, d2, d3 = torch.utils.data.random_split(data, (train_size, val_size, test_size),
                                                   generator=torch.Generator().manual_seed(seed))
        if dataset_type == 'train':
            self.reserved_dataset = d1
        elif dataset_type == 'val':
            self.reserved_dataset = d2
        elif dataset_type == 'test':
            self.reserved_dataset = d3
        else:
            raise AttributeError()

        self.dilation = dilation
        self.step = step

    def normalize(self, data):
        mean = np.mean(data, axis=(0, 1))
        std = np.std(data, axis=(0, 1))
        return (data - mean) / std, mean, std

    def __len__(self):
        return len(self.reserved_dataset)

    def __getitem__(self, item):
        data_tuple = self.reserved_dataset.__getitem__(item)
        # data_tuple = self.reserved_data[item * self.step]
        data_in, data_out = [data_tuple[:, self.io[_]] for _ in range(2)]

        return data_in, data_out


class CTSample:
    def __init__(self, sp: float, base_tp=0.1, evenly=False):
        self.sp = np.clip(sp, 0.01, 1.0)
        self.base_tp = base_tp
        self.evenly = evenly

    def batch_collate_fn(self, batch):

        external_input, observation = [torch.from_numpy(np.stack(x)) for x in zip(*batch)]
        bs, l, _ = external_input.shape
        time_steps = torch.arange(external_input.size(1)) * self.base_tp
        data = torch.cat([external_input, observation], dim=-1)
        new_data, tp = subsample_indexes(data, time_steps, self.sp, evenly=self.evenly)
        external_input, observation = new_data[..., :external_input.shape[-1]], new_data[..., -observation.shape[-1]:]

        # region [ati, t_{i} - t_{i-1}]
        # tp = torch.cat([tp[..., 0:1], tp], dim=-1)
        # dt = tp[..., 1:] - tp[..., :-1]
        # endregion

        # region [ati, t_{i+1} - t_{i}]
        tp = torch.cat([tp, tp[..., -1:]], dim=-1)
        dt = tp[..., 1:] - tp[..., :-1]
        # endregion

        def add_tp(x, tp):
            return torch.cat([
                x,
                tp.repeat(bs, 1).unsqueeze(dim=-1)
            ], dim=-1)

        external_input = add_tp(external_input, dt)
        # observation = add_tp(observation, dt)
        return external_input, observation


# PR-SSM Dataset: actuator, ballbeam, drive, dryer, gas_furnace
class ActuatorDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        begin_pos = []

        # 每个column对应的数据含义 ['in','out']
        self.df = df
        self.used_columns = ['u', 'p']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df['u'], dtype=np.float32)
        data_out = np.array(data_df['p'], dtype=np.float32)

        return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)


class BallbeamDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        begin_pos = []

        # 每个column对应的数据含义 ['in','out']
        self.df = df
        self.used_columns = ['0', '1']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df['0'], dtype=np.float32)
        data_out = np.array(data_df['1'], dtype=np.float32)

        return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)


class DriveDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        begin_pos = []

        # 每个column对应的数据含义 ['in','out']
        self.df = df
        self.used_columns = ['u1', 'z1']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df['u1'], dtype=np.float32)
        data_out = np.array(data_df['z1'], dtype=np.float32)

        return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)


class DryerDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        begin_pos = []

        # 每个column对应的数据含义 ['in','out']
        self.df = df
        self.used_columns = ['0', '1']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df['0'], dtype=np.float32)
        data_out = np.array(data_df['1'], dtype=np.float32)

        return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)


class GasFurnaceDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        begin_pos = []

        # 每个column对应的数据含义 ['in','out']
        self.df = df
        self.used_columns = ['0', '1']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df['0'], dtype=np.float32)
        data_out = np.array(data_df['1'], dtype=np.float32)

        return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)


class SarcosArmDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        begin_pos = []

        # 每个column对应的数据含义 ['in','out']
        self.df = df
        self.used_columns = ['0', '1', '2', '3', '4', '5', '6',
                             '21', '22', '23', '24', '25', '26', '27']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df[['21', '22', '23', '24', '25', '26', '27']], dtype=np.float32)
        data_out = np.array(data_df[['0', '1', '2', '3', '4', '5', '6']], dtype=np.float32)

        return data_in, data_out

