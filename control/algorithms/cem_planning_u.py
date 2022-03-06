#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
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

    def __init__(self, set_value, input_dim, output_dim, init_action, length, num_samples=32, max_iters=50, device='cpu', time=0):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.length = length
        self.num_samples = num_samples
        self.device = device
        self.max_iters = max_iters
        '''
        # 模型训练数据集的归一化参数
        df_scale = pd.read_csv(os.path.join(os.getcwd(), "control/df_scale.csv"), encoding='utf-8').values
        self.s_mean = np.array(df_scale[:, 1], dtype='float64')
        self.s_std = np.array(df_scale[:, 2], dtype='float64')
        self.scale = Scale(self.s_mean, self.s_std)
        self.set_value = np.array([55, 74, 236.9, 74, 100], dtype=np.float32)  # 设定值
        '''

        self.set_value = np.array(set_value)
        self.figs_path = os.path.join(os.getcwd(), 'control/figs')
        if not os.path.exists(self.figs_path):
            os.makedirs(self.figs_path)
        self.test = False
        self.cem_time = int(time)
        print("cem init over")
        self.last_action = init_action

    def eval(self, obs, action):
        """

        Args:
            new_state: with shape(samples, length, output_dim)
            action: with shape (samples, length, input_dim)

        Returns: cost shape with (samples, 1)
        """
        # 设定值 obs_set 正常为在cem_planning() 的request 中获取， 测试先固定设定值：{pressure:250, v_in:236.9, v_out:236.9, c_in: 74, c_out = 74}
        # obs_set_value = {'pressure':250, 'c_out': 74, 'v_in':236.9, 'c_in': 74}  # 未归一化 应为一个和obs_pred_j shape相同的一个tensor
        # action_set_value = {'v_out':236.9}
        MSE = torch.nn.MSELoss(reduction='none')

        '''
        # 归一化
        set_value_scale = self.scale.scale_array(np.array(self.set_value))
        set_value_scale = set_value_scale.tolist()
        '''
        obs_set_value = torch.Tensor(self.set_value.tolist()[:self.output_dim])
        obs_set_value = obs_set_value.expand(self.num_samples, self.length, self.output_dim).to(self.device).contiguous()  #
        obs = obs.permute(2, 0, 1).contiguous()
        obs_set_value = obs_set_value.permute(2, 0, 1).contiguous()
        # TODO: 不许出现for循环
        mse_list = [MSE(obs_set_value[x].unsqueeze(-1), obs[x].unsqueeze(-1)).mean(dim=1) for x in range(0, self.output_dim)]
        loss = 0
        # for i in range(0,self.output_dim):
        #     loss = loss + mse_list[i]
        # loss_j = loss                        # (num_sample, 1)

        loss_j = mse_list[-1]  # 仅使用reward作为控制指标（内含f和c）

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
            for i in range(self.max_iters):
                """
                1. 从last_seq_distribution采样
                2. 优化last_seq_distribution
                """
                u_j = last_seq_distribution.sample((self.num_samples, ))     # tensor.shape
                # 针对IB系统对action_j进行处理
                # self.state['v'] = np.clip(self.state['v'] + delta[0], 0., 100.)
                # self.state['g'] = np.clip(self.state['g'] + 10 * delta[1], 0., 100.)
                # self.state['h'] = np.clip(
                #     self.state['h'] + ((self.maxRequiredStep / 0.9) * 100. / self.gsScale) * delta[2], 0., 100.)
                # 获取last action_j # TODO: 兼容性问题，IB的action不是单纯加上delta_u，目前为针对性组合
                action_j = torch.from_numpy(np.array(self.last_action)).expand(u_j.shape).contiguous().to(
                    self.device)  # last_action 扩维
                action_j = action_j.permute(2, 0, 1).contiguous()
                u_j = u_j.permute(2, 0, 1).contiguous()
                action_j[0] = torch.clamp(action_j[0] + u_j[0], 0., 100.)
                action_j[1] = torch.clamp(action_j[1] + 10 * u_j[1], 0., 100.)
                action_j[2] = torch.clamp(
                    action_j[2] + ((np.sin(15./180.*np.pi) / 0.9) * 100. / 2.* 1.5 + 100.* 0.02) * u_j[2], 0., 100.)
                action_j = action_j.permute(1, 2, 0).contiguous()
                u_j = u_j.permute(1, 2, 0).contiguous()

                # action_j归一化   # TODO: 兼容性问题，model.mean需要兼容不同系统，后续修改为对应action_name的mean.values
                aj_mean = torch.from_numpy(np.array(self.model.mean.values.tolist()[1:4])).expand(action_j.shape).contiguous().to(self.device)
                aj_std = torch.from_numpy(np.array(self.model.std.values.tolist()[1:4])).expand(action_j.shape).contiguous().to(self.device)
                action_j = ( action_j - aj_mean ) / aj_std
                # action_j变维
                action_j = action_j.permute(1, 0, 2).contiguous()      # model.forward_prediction(([len, batch_size, input_size]), memory_state)
                # TODO : 不同的模型memory_state定义不同，需要兼容
                memory_state['hn'] = memory_state['hn'].expand(self.num_samples, memory_state['hn'].shape[-1]).contiguous()
                memory_state['rnn_hidden'] = memory_state['rnn_hidden'].expand(memory_state['rnn_hidden'].shape[0], self.num_samples, memory_state['rnn_hidden'].shape[-1]).contiguous()
                output, _ = model.forward_prediction(action_j, max_prob=True, memory_state=memory_state)
                obs_pred_j = output['predicted_seq']
                obs_pred_j = obs_pred_j.permute(1, 0, 2).contiguous()  # (J, length, output_dim)
                # action_j = action_j.permute(1, 0, 2).contiguous()
                reward_j = self.eval(obs_pred_j, u_j)
                reward_j = reward_j.squeeze(1)
                _, K = reward_j.sort()
                K = K[0:k]
                u_k = u_j[K]
                mean = u_k.mean(dim=0)
                scale = u_k.std(dim=0)
                new_seq_distribution = Normal(mean, scale)
                # TODO: 不许出现for循环，直接action_k[0].mean(dim=-2)就行
                u_f = [u_k[0].permute(1, 0)[x].mean() for x in range(0, self.input_dim)]
                action_f = self.last_action
                action_f[0] = torch.clamp(action_f[0] + u_f[0], 0., 100.)
                action_f[1] = torch.clamp(action_f[1] + 10 * u_f[1], 0., 100.)
                action_f[2] = torch.clamp(
                    action_f[2] + ((np.sin(15. / 180. * np.pi) / 0.9) * 100. / 2. * 1.5 + 100. * 0.02) * u_f[2], 0.,
                    100.)
                self.last_action = action_f                 # 更新上一时刻的系统动作
                '''反归一化模块
                action_list.append(float(action_k[0].mean())*self.s_std[-1]+self.s_mean[-1])   # 反归一化
                pressure_list.append(float(obs_k[0][0].mean())*self.s_std[0]+self.s_mean[0])  # pressure 应该是action_k[0].mean()预测出的
                '''
                # print("iter = ", i)
                pass

            print("CEM over")

            return new_seq_distribution, action_f, u_f



