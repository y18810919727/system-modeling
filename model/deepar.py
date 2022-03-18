#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
from model.common import DBlock, PreProcess, MLP
# from model.common import
from model.common import DiagMultivariateNormal as MultivariateNormal

from common import softplus, inverse_softplus, cov2logsigma, logsigma2cov, split_first_dim, merge_first_two_dims
from model.func import normal_differential_sample, multivariate_normal_kl_loss
import time
from common import TimeRecorder


class DeepAR(nn.Module):

    def __init__(self, input_size, observations_size, net_type='rnn', k=16, num_layers=1, dropout=0.1):

        super(DeepAR, self).__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = k
        self.observations_size = observations_size
        self.drop_out = dropout

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bias=True,
                            batch_first=False,
                            dropout=self.drop_out)

        # 初始化LSTM忘记门偏置为1
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        # 两个线性层解码高斯分布均值、标准差
        self.distribution_mu = nn.Linear(self.hidden_size * num_layers, observations_size)
        self.distribution_presigma = nn.Linear(self.hidden_size * num_layers, observations_size)

    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
        """

        Args:
            external_input_seq: 输入序列 (len, batch_size, input_size)
            observations_seq: 观测序列 (len, batch_size, ob_size)
            memory_state: 字典，模型记忆

        Returns: 隐变量均值,隐变量方差(确定性隐变量可以将其置0), 新的memory_state

        把其他需要计算loss的变量存在对象里

        """
        l, batch_size, _ = external_input_seq.size()

        hidden, cell = (
        torch.zeros(self.num_layers, batch_size, self.hidden_size, device=external_input_seq.device),
        torch.zeros(self.num_layers, batch_size, self.hidden_size,
                    device=external_input_seq.device)) if memory_state is None else (
            memory_state['hidden'], memory_state['cell'])
        state_mu = []
        state_logsigma = []
        for t in range(l):
            output, (hidden, cell) = self.lstm(torch.unsqueeze(external_input_seq[t], dim=0), (hidden, cell))
            hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
            sigma = self.distribution_presigma(hidden_permute)
            mu = self.distribution_mu(hidden_permute)
            state_mu.append(mu)
            state_logsigma.append(sigma)

        outputs = {
            'state_mu': torch.stack(state_mu, dim=0),
            'state_logsigma': torch.stack(state_logsigma, dim=0),
        }
        memory_state = {
            'hidden': hidden,
            'cell': cell,
        }
        return outputs, memory_state

    def forward_prediction(self, external_input_seq, n_traj, memory_state=None):
        """

        Args:
            n_traj:
            external_input_seq: 输入序列 (len, batch_size, input_size)
            max_prob: 如果为False，从预测分布中随机采样，如果为True ,返回概率密度最大的估计值
            memory_state: 字典，模型记忆

        Returns: 元祖 (
            预测的观测序列分布 torch.distributions.MultivariateNormal,
            预测序列(序列怎么得到与max_prob有关) shape-> (len,batch_size,ob_size),
            模型记忆_ 字典
            )

        """
        l, batch_size, _ = external_input_seq.size()
        hidden, cell = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=external_input_seq.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size,
                        device=external_input_seq.device)) if memory_state is None else (
            memory_state['hidden'], memory_state['cell'])

        predicted_seq_sample = []
        external_input_seq = external_input_seq.repeat(1, n_traj, 1)
        hidden = hidden.repeat(1, n_traj, 1)
        cell = cell.repeat(1, n_traj, 1)

        with torch.no_grad():
            for t in range(l):
                output, (hidden, cell) = self.lstm(torch.unsqueeze(external_input_seq[t], dim=0), (hidden, cell))
                hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
                sigma = self.distribution_presigma(hidden_permute)
                mu = self.distribution_mu(hidden_permute)
                # state_mu.append(mu)
                # state_logsigma.append(sigma)
                observations_dist = self.decode_observation(
                    {'state_mu': mu, 'state_logsigma': sigma},
                    mode='dist'
                )
                observations_sample = split_first_dim(
                    normal_differential_sample(observations_dist),
                    (n_traj, batch_size)
                )
                observations_sample = observations_sample.permute(1, 0, 2)
                predicted_seq_sample.append(observations_sample)

        predicted_seq_sample = torch.stack(predicted_seq_sample, dim=0)
        predicted_seq = torch.mean(predicted_seq_sample, dim=2)
        predicted_dist = MultivariateNormal(
            predicted_seq_sample.mean(dim=2), torch.diag_embed(predicted_seq_sample.var(dim=2))
        )

        memory_state = {
            'hidden': hidden,
            'cell': cell
        }
        outputs = {
            'predicted_seq_sample': predicted_seq_sample,
            'predicted_dist': predicted_dist,
            'predicted_seq': predicted_seq

        }
        return outputs, memory_state

    def call_loss(self, external_input_seq, observations_seq, memory_state=None):
        """
        Args:
            mu: 高斯似然均值
            sigma: 高斯似然方差
        Returns:
            三个标量: loss, kl_loss, decoding_loss， 没有后面两部分用0站位
            loss: 负对数似然loss
        loss要在batch_size纬度上取平均

        方法调用时不会给额外的输入参数，需在每次forward_prediction和forward_posterior之后将解码所需的信息存储在self里

        deepAR中会考虑不存在的label值，这里不考虑
        """
        outputs, memory_state = self.forward_posterior(external_input_seq, observations_seq, memory_state)

        state_mu = outputs['state_mu']
        state_logsigma = outputs['state_logsigma']
        l, batch_size, _ = observations_seq.shape
        distribution = MultivariateNormal(state_mu, logsigma2cov(state_logsigma))
        likelihood = distribution.log_prob(observations_seq)
        loss_batch_mean = torch.mean(likelihood, dim=1)
        loss = -torch.squeeze(torch.sum(loss_batch_mean, dim=0))
        return {
            'loss': loss,
            'kl_loss': 0,
            'likelihood_loss':0
        }

    def decode_observation(self, outputs, mode='sample'):
        """

        Args:
            mode: dist or sample

        Returns:
            model为sample时，从分布采样(len,batch_size,observation)
            为dist时，直接返回分布对象torch.distributions.MultivariateNormal

        方法调用时不会给额外的输入参数，需在每次forward_prediction和forward_posterior之后将解码所需的信息存储在self里
        """
        observations_normal_dist = torch.distributions.MultivariateNormal(
            outputs['state_mu'], logsigma2cov(outputs['state_logsigma'])
        )
        if mode == 'dist':
            return observations_normal_dist
        elif mode == 'sample':
            return observations_normal_dist.sample()
