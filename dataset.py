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
        from data.fake_data_generator import begin_state_sigma, begin_state_mu

        return external_input, np.expand_dims(observation, axis=1), np.expand_dims(state, axis=1), np.array(
            [begin_state_mu], dtype=np.float32), np.array(
            [begin_state_sigma], dtype=np.float32)

    def __len__(self):
        return len(self.data)


class WesternDataset(Dataset):
    def __init__(self,df_list, length=1000):
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
            for j in range(0,df.shape[0]-length+1, length//10):
                begin_pos_pair.append((i,j))
        self.begin_pos_pair = begin_pos_pair
        self.df_split_all = df_split_all

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
                c_in*v_in - c_out*v_out
            ],
            axis=1)
        observation = pressure

        begin_state_sigma = float(3.0)
        begin_state_mu = float(pressure[0]/0.8)

        # 真实数据不知道干砂质量，这快就随便写了
        state = pressure/0.80

        return external_input, np.expand_dims(observation, axis=1), np.expand_dims(state, axis=1), np.array(
            [begin_state_mu], dtype=np.float32), np.array(
            [begin_state_sigma], dtype=np.float32)


