#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch

from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

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
        c_in = self.data[item][:,self.names.index('c_in')]
        v_in = self.data[item][:,self.names.index('v_in')]
        c_out = self.data[item][:,self.names.index('c_out')]
        v_out = self.data[item][:,self.names.index('v_out')]
        pressure = self.data[item][:,self.names.index('pressure')]
        mass = self.data[item][:,self.names.index('mass')]
        external_input = np.stack(
            [
                c_in*c_in*c_in*v_in - c_out*c_out*c_out*v_out,
                c_in*c_in*v_in - c_out*c_out*v_out,
                c_in*v_in - c_out*v_out
            ],
            axis=1)
        observation = pressure
        state = mass

        return external_input, np.expand_dims(observation, axis=1)

    def __len__(self):
        return len(self.data)


class WesternDataset(Dataset):
    def __init__(self,df_list, length=1000, step=5):
        if not isinstance(df_list, list):
            df_list = [df_list]
        df_split_all = []
        begin_pos_pair = []

        #每个column对应的数据含义 ['c_in','c_out', 'v_out', 'v_in', 'pressure']
        self.used_columns = ['4','11','14','16','17']
        self.length = length

        for df in df_list:
            df_split_all = df_split_all + self.split_df(df[self.used_columns])
        for i, df in enumerate(df_split_all):
            for j in range(0,df.shape[0]-length+1, step):
                begin_pos_pair.append((i,j))
        self.begin_pos_pair = begin_pos_pair
        self.df_split_all = df_split_all
        self.df_split_all = self.normalize(self.df_split_all)

    def normalize(self, df_all_list):
        df_all = df_all_list[0].append(df_all_list[1:], ignore_index=True)
        mean = df_all.mean()
        std = df_all.std()
        return [(df-mean)/std for df in df_all_list]

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
        split_indexes = [-1]+split_indexes + [df.shape[0]]
        for i in range(len(split_indexes)-1):
            if split_indexes[i+1]-split_indexes[i]-1<self.length:
                continue

            new_df = df.iloc[split_indexes[i]+1:split_indexes[i+1]]
            assert new_df.isnull().sum().sum() == 0
            df_list.append(new_df)
        return df_list

    def __len__(self):
        return len(self.begin_pos_pair)

    def __getitem__(self, item):
        df_index, pos = self.begin_pos_pair[item]
        data_array = np.array(self.df_split_all[df_index].iloc[pos:pos+self.length], dtype=np.float32)
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        c_in, c_out, v_out, v_in, pressure = [np.squeeze(x, axis=1) for x in np.hsplit(data_array, 5)]

        v_in = v_in*0.05
        v_out = v_out*0.05

        external_input = np.stack(
            [
                c_in*c_in*c_in*v_in - c_out*c_out*c_out*v_out,
                c_in*c_in*v_in - c_out*c_out*v_out,
                c_in*v_in - c_out*v_out,
                v_in - v_out,
            ],
            axis=1)
        observation = pressure


        # 真实数据不知道干砂质量，这里就随便写了
        state = pressure/0.80

        return external_input, np.expand_dims(observation, axis=1)


class CstrDataset(Dataset):
    def __init__(self, df, length=1000, step=5):

        df_split_all = []
        begin_pos = []

        #每个column对应的数据含义 ['in','out1', 'out2']
        self.df = df
        self.used_columns = ['0','1','2']
        self.length = length
        for j in range(0, df.shape[0]-length+1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df-mean)/std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos+self.length]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        data_in = np.array(data_df['0'], dtype=np.float32)
        data_out = np.array(data_df[['1', '2']], dtype=np.float32)

        #return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)
        return np.expand_dims(data_in, axis=1), data_out


class WindingDataset(Dataset):
    def __init__(self, df, length=1000, step=5):

        df_split_all = []
        begin_pos = []

        #每个column对应的数据含义 ['in','out1', 'out2']
        self.df = df
        self.used_columns = ['0', '1', '2', '3', '4', '5', '6']
        self.length = length
        for j in range(0, df.shape[0]-length+1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df-mean)/std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos+self.length]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        data_in = np.array(data_df[['0','1','2','3','4']], dtype=np.float32)
        data_out = np.array(data_df[['5', '6']], dtype=np.float32)

        #return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)
        return data_in, data_out

