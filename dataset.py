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
        self.group =  defaultdict(list)
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
            begin_state_mu, dtype=np.float32), np.array(
            begin_state_sigma, dtype=np.float32)

    def __len__(self):
        return len(self.data)

