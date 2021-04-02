#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import random


class ThickenerPressureSimulation:

    def __init__(self,
                 model,
                 dataset_path,
                 random_seed=None,
                 ):

        """
        :param model: 建模实验中保存的模型
        :param random_seed: 随机化种子，会决定浓密机的初始状态和
        """
        self.dataset_path = dataset_path
        self.model = model

        if random_seed is None:
            random_seed = random.randint(0, 1000000)
            print('random_seed:' + str(random_seed))
        self.random_seed = random_seed

    def step(self, v_out_new: float):

        """

        1. 利用model预测下一时刻的pressure, c_out
        2. 从dataset取出新的v_in, v_out
        3. 更新浓密机hidden_state
        4. 返回五元组

        Args:
            v_out_new:

        Returns:

        """

        return self.get_current_state()

    def get_current_state(self):

        return {
            'pressure': 0.0,
            'v_in': 0.0,
            'v_out': 0.0,
            'c_in': 0.0,
            'c_out': 0.0
        }

