#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import os
import torch

#####
from model.func import normal_differential_sample
from model.common import DiagMultivariateNormal as MultivariateNormal
#####


class PIDPlanning:

    def __init__(self, dt, max, min, KP, KI, KD, time=0):

        # self.input_dim = input_dim
        # self.output_dim = output_dim
        # self.length = length
        # self.num_samples = num_samples
        # self.device = device
        # self.max_iters = max_iters
        self.dt = dt
        self.max = max
        self.min = min
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.sum_err = 0
        self.last_err = 0

        self.figs_path = os.path.join(os.getcwd(), 'control/pid_figs')
        if not os.path.exists(self.figs_path):
            os.makedirs(self.figs_path)
        self.test = False
        self.pid_time = int(time)

        print("pid init over")

    def solve(self, setPoint, pv=None):
        """

        Args:
            setPoint: 设定值
            pv: process value 过程值

        Returns:

        """

        if pv is None:
            pv = 50

        if self.test:
            return 0
        else:
            exp_val_c = setPoint
            now_val_c = pv

            error = exp_val_c - now_val_c  # 误差
            p_out = self.KP * error  # 比例项
            self.sum_err += error * self.dt
            i_out = self.Ki * self.sum_err  # 积分项
            derivative = (error - self.last_err) / self.dt  # 微分项
            d_out = self.KD * derivative

            output = p_out + i_out + d_out

            if output > self.max:
                output = self.max
            elif output < self.min:
                output = self.min

            self.last_err = self.now_err

        return output