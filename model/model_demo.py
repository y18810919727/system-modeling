#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
from model.common import DBlock, PreProcess, MLP

from common import softplus, inverse_softplus, cov2logsigma, logsigma2cov, split_first_dim, merge_first_two_dims
from model.func import normal_differential_sample, multivariat_normal_kl_loss

class DeepAR(nn.Module):

    def __init__(self, input_size, state_size, observations_size, net_type='rnn', k=16, num_layers=1):

        super(DeepAR, self).__init__()


    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
        """

        Args:
            external_input_seq: 输入序列 (len, batch_size, input_size)
            observations_seq: 观测序列 (len, batch_size, ob_size)
            memory_state: 字典，模型记忆

        Returns: 隐变量均值,隐变量方差(确定性隐变量可以将其置0), 新的memory_state

        把其他需要计算loss的变量存在对象里

        """

        self.memory_state = memory_state
        self.external_input_seq = external_input_seq
        self.observations_seq = observations_seq

        l, batch_size, _ = external_input_seq.size()

        return self.state_mu, self.state_logsigma, memory_state

    def forward_prediction(self, external_input_seq, max_prob=False, memory_state=None):
        """

        Args:
            external_input_seq: 输入序列 (len, batch_size, input_size)
            max_prob: 如果为False，从预测分布中随机采样，如果为True ,返回概率密度最大的估计值
            memory_state: 字典，模型记忆

        Returns: 元祖 (
            预测的观测序列分布 torch.distributions.MultivariateNormal,
            预测序列(序列怎么得到与max_prob有关) shape-> (len,batch_size,ob_size),
            模型记忆_ 字典
            )

        """
        self.memory_state = memory_state
        self.external_input_seq = external_input_seq

        l, batch_size, _ = external_input_seq.size()


        # return observations_dist, observations_sample, {'hn': h, 'rnn_hidden': rnn_hidden_state}

    def call_loss(self):
        """
        Args:

        Returns:
            三个标量: loss, kl_loss, decoding_loss， 没有后面两部分用0站位

        loss要在batch_size纬度上取平均

        方法调用时不会给额外的输入参数，需在每次forward_prediction和forward_posterior之后将解码所需的信息存储在self里

        """


        # return (kl_sum - generative_likelihood)/batch_size, kl_sum/batch_size, -generative_likelihood/batch_size

    def decode_observation(self, mode='sample'):
        """

        Args:
            mode: dist or sample

        Returns:
            model为sample时，从分布采样(len,batch_size,observation)
            为dist时，直接返回分布对象torch.distributions.MultivariateNormal

        方法调用时不会给额外的输入参数，需在每次forward_prediction和forward_posterior之后将解码所需的信息存储在self里
        """
        # mean, logsigma = self.decoder(
        #     torch.cat([self.process_z(self.sampled_state), self.h_seq[:-1]], dim=-1)
        # )
        # observations_normal_dist = torch.distributions.MultivariateNormal(
        #     mean, logsigma2cov(logsigma)
        # )
        # if mode == 'dist':
        #     return observations_normal_dist
        # elif mode == 'sample':
        #     return observations_normal_dist.sample()
