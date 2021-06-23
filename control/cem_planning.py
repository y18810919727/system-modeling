#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import numpy as np
import pandas as pd
import copy
from torch.distributions.normal import Normal
from control.thickener_pressure_simulation import ThickenerPressureSimulation
from control.scale import Scale
from matplotlib import pyplot as plt
#####
from model.func import normal_differential_sample
from model.common import DiagMultivariateNormal as MultivariateNormal
#####
class CEMPlanning:

    def __init__(self, input_dim, output_dim, length, num_samples=32, max_iters=50, device='cpu', time=0):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.length = length      # 需不需要有一定的长度？
        self.num_samples = num_samples
        self.device = device
        self.max_iters = max_iters

        # scale = Scale(mean=np.zeros(5), std=np.ones(5))
        # 模型训练数据集的归一化参数
        df_scale = pd.read_csv(os.path.join(os.getcwd(), "control/df_scale.csv"), encoding='utf-8').values
        self.s_mean = np.array(df_scale[:, 1], dtype='float64')
        self.s_std = np.array(df_scale[:, 2], dtype='float64')
        self.scale = Scale(self.s_mean, self.s_std)
        self.set_value = np.array([55, 74, 236.9, 74, 100], dtype=np.float32)  # 设定值
        self.figs_path = os.path.join(os.getcwd(), 'control/figs')
        if not os.path.exists(self.figs_path):
            os.makedirs(self.figs_path)
        self.test = False
        self.cem_time = int(time)
        print("cem init over")

    def eval(self, state, action):
        """

        Args:
            new_state: with shape(samples, length, output_dim)
            action: with shape (samples, length, input_dim)

        Returns: cost shape with (samples, 1)
        """
        # RSM(xt-x*)2
        # RRSE/ MSE
        # 设定值 obs_set 正常为在cem_planning() 的request 中获取， 测试先固定设定值：{pressure:250, v_in:236.9, v_out:236.9, c_in: 74, c_out = 74}
        # obs_set_value = {'pressure':250, 'c_out': 74, 'v_in':236.9, 'c_in': 74}  # 未归一化 应为一个和obs_pred_j shape相同的一个tensor
        # action_set_value = {'v_out':236.9}
        MSE = torch.nn.MSELoss(reduction='none')

        # 归一化
        set_value_scale = self.scale.scale_array(np.array(self.set_value))
        set_value_scale = set_value_scale.tolist()
        # v_out 为最后一个数
        obs_set_value = set_value_scale[:-1]
        pressure_set = torch.Tensor([obs_set_value[0]])
        c_out_set = torch.Tensor([obs_set_value[1]])
        pressure_set = pressure_set.expand(self.num_samples, self.length, self.input_dim).to(self.device).contiguous()
        c_out_set = c_out_set.expand(self.num_samples, self.length, self.input_dim).to(self.device).contiguous()
        mse_pressure = MSE(pressure_set, state.permute(2, 0, 1).contiguous()[0].unsqueeze(-1)).mean(dim=1)
        mse_c_out = MSE(c_out_set, state.permute(2, 0, 1).contiguous()[1].unsqueeze(-1)).mean(dim=1)
        # 统计参数1 observation预测和设定值的RRSE      (samples, output_dim = 4)
        #obs_pred_rrse = CEM_RRSE(obs_set, state)
        # 统计参数2 对action 的限制---上下界（设定值）   (samples, input_dim = 1)
        #action_j_rrse = CEM_RRSE(action_set, action)
        loss_j = mse_pressure   # mse_c_out   # (num_sample, 1)

        return loss_j

    def solve(self, model, memory_state, last_seq_distribution=None):
        """

        Args:
            model:
            last_seq_distribution:   # 对角高斯分布
            memory_state:

        Returns:

            new_seq_distribution: An normal distribution with shape [length, ]


        """
        if last_seq_distribution is None:
            mean = torch.zeros((self.length, self.input_dim)).to(self.device)
            scale = torch.ones((self.length, self.input_dim)).to(self.device)
        else:
            mean_last = torch.zeros((1, self.input_dim)).to(self.device)
            scale_last = torch.ones((1, self.input_dim)).to(self.device)
            mean = torch.cat([last_seq_distribution.mean[1:], mean_last], dim=0)
            scale = torch.cat([last_seq_distribution.scale[1:], scale_last], dim=0)
        last_seq_distribution = Normal(mean, scale)  # Normal  (self.length, self.input_dim)

        if self.test:
            action_test = torch.tensor(1, dtype=float, device=self.device)
            new_seq_distribution = None
            return new_seq_distribution, action_test
        else:

            k = 10
            action_list = []
            pressure_list = []
            for i in range(self.max_iters):      # 中间的迭代除了更新last_seq_distribution还有没有别的作用
                """
                1. 从last_seq_distribution采样
                2. 优化last_seq_distribution
                """
                action_j = last_seq_distribution.sample((self.num_samples,))
                action_j = action_j.permute(1, 0, 2).contiguous()   # model.forward_prediction(([len, batch_size, input_size]), memory_state)
                memory_state['hn'] = memory_state['hn'].expand(self.num_samples, memory_state['hn'].shape[-1]).contiguous()
                memory_state['rnn_hidden'] = memory_state['rnn_hidden'].expand(memory_state['rnn_hidden'].shape[0], self.num_samples, memory_state['rnn_hidden'].shape[-1]).contiguous()
                obs_pred_dist, obs_pred_j, _ = model.forward_prediction(action_j, max_prob=True, memory_state=memory_state)
                obs_pred_j = obs_pred_j.permute(1, 0, 2).contiguous()  # (J, length, output_dim)
                action_j = action_j.permute(1, 0, 2).contiguous()
                reward_j = self.eval(obs_pred_j, action_j)

                reward_j = reward_j.squeeze(1)
                _, K = reward_j.sort()
                K = K[0:k]
                # action_k = []
                # obs_k = []
                action_k = action_j[K]
                obs_k = obs_pred_j[K]
                mean = action_k.mean(dim=0)
                scale = action_k.std(dim=0)
                new_seq_distribution = Normal(mean, scale)
                action_f = action_k[0].mean()
                action_list.append(float(action_k[0].mean())*self.s_std[-1]+self.s_mean[-1])   # 反归一化
                pressure_list.append(float(obs_k[0][0].mean())*self.s_std[0]+self.s_mean[0])  # pressure 应该是action_k[0].mean()预测出的
                # print("iter = ", i)
                pass  # ？

            # 保存 仿真的当前时刻浓密机参数状态----保存到csv文件中

            print("CEM over")


            return new_seq_distribution, action_f



