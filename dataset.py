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


class FakeDataset(Dataset):

    def __init__(self, df):
        """
        round,t,mass,pressure,v_in,c_in,v_out,c_out
        :param df:
        """
        self.names = list(df.columns)[3:]
        data = np.array(df)
        self.group = defaultdict(list)
        self.data = {}
        for x in data:
            self.group[int(x[1])].append(x[3:])
        for key in self.group.keys():
            self.data[key] = np.array(self.group[key], dtype=np.float32)
        del self.group

    def __getitem__(self, item):
        c_in = self.data[item][:, self.names.index('c_in')]
        v_in = self.data[item][:, self.names.index('v_in')]
        c_out = self.data[item][:, self.names.index('c_out')]
        v_out = self.data[item][:, self.names.index('v_out')]
        pressure = self.data[item][:, self.names.index('pressure')]
        mass = self.data[item][:, self.names.index('mass')]
        external_input = np.stack(
            [
                c_in * c_in * c_in * v_in - c_out * c_out * c_out * v_out,
                c_in * c_in * v_in - c_out * c_out * v_out,
                c_in * v_in - c_out * v_out
            ],
            axis=1)
        observation = pressure
        state = mass

        return external_input, np.expand_dims(observation, axis=1)

    def __len__(self):
        return len(self.data)


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
        data_in = np.array(data_df[['delta_v','delta_g','delta_h']], dtype=np.float32)
        data_out = np.array(data_df[['v','g','h','f', 'c', 'reward']], dtype=np.float32)
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


class CTSample:
    def __init__(self, sp: float, base_tp=0.1):
        self.sp = np.clip(sp, 0.01, 1.0)
        self.base_tp = base_tp

    def batch_collate_fn(self, batch):

        external_input, observation = [torch.from_numpy(np.stack(x)) for x in zip(*batch)]
        bs, l, _ = external_input.shape
        time_steps = torch.arange(external_input.size(1)) * self.base_tp
        data = torch.cat([external_input, observation], dim=-1)
        new_data, tp = subsample_indexes(data, time_steps, self.sp)
        tp = torch.cat([tp[..., 0:1], tp], dim=-1)
        dt = tp[..., 1:] - tp[..., :-1]
        external_input, observation = new_data[..., :external_input.shape[-1]], new_data[..., -observation.shape[-1]:]

        def add_tp(x, tp):
            return torch.cat([
                x,
                tp.repeat(bs, 1).unsqueeze(dim=-1)
            ], dim=-1)

        external_input = add_tp(external_input, dt)
        # observation = add_tp(observation, dt)
        return external_input, observation
