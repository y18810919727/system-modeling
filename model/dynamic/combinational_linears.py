#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch


import torch
from torch import nn
from model.func import weighted_linear, normal_differential_sample
from common import logsigma2cov
from torch.distributions import MultivariateNormal


class CombinationalLinears(nn.Module):

    def __init__(self, input_size, state_size, num_linears=8, num_layers=1, hidden_size=32):
        super(CombinationalLinears, self).__init__()
        # In combinated linears, learnable linear matrix are weighted upon a softmax output.
        self.num_linears = num_linears
        self.state_size = state_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linears_input = nn.ModuleList([torch.nn.Linear(input_size, state_size) for _ in range(num_linears)])
        self.linears_state = nn.ModuleList([torch.nn.Linear(state_size, state_size) for _ in range(num_linears)])

        self.linears_weight_lstm = nn.LSTM(state_size, hidden_size, num_layers)
        # weight lstm的初始隐状态可学习
        self.weight_hidden_state0 = torch.nn.Parameter(torch.zeros(hidden_size))
        self.linears_weight_fcn = nn.Sequential(
            nn.Linear(hidden_size, num_linears),
            nn.Softmax(dim=-1)
        )
        self.estimate_logsigma = torch.nn.parameter.Parameter(torch.randn(num_linears, state_size))

    def forward(self, state, external_input, args):
        """
        Args:
            state: shape ( ..., state_size)
            external_input: (..., input_size)
            args: tuple(lstm state for weight adaption, ), each state has the shape (ls, bs, num_layers, hidden_size)
            or (bs, num_layers, hidden_size)

        Returns: (distribution of new state, new lstm state)

        """
        assert state.size()[-1] == self.state_size and external_input.size()[-1] == self.input_size

        (lstm_weight_hidden_state), = args

        try:
            weight_map = self.linears_weight_fcn(
                lstm_weight_hidden_state[0][-1]
            )  # (bs, num_layers)
            next_mu = weighted_linear(
                state, linears=self.linears_state, weight=weight_map
            ) + weighted_linear(external_input, linears=self.linears_input, weight=weight_map)

            next_cov = logsigma2cov(
                weight_map @ self.estimate_logsigma
            )
        except Exception as e:
            print('external_input shape: %s    state shape: %s\n' % (external_input.shape, state.shape))
            raise e

        predicted_dist = MultivariateNormal(next_mu, next_cov)
        sample = normal_differential_sample(predicted_dist)

        _, new_lstm_weight_hidden_state = self.linears_weight_lstm(sample.unsqueeze(dim=0),
                                               lstm_weight_hidden_state)

        return MultivariateNormal(next_mu, next_cov), sample, (new_lstm_weight_hidden_state, weight_map)
        # 采样i时刻隐变量
        # state = normal_differential_sample(torch.distributions.MultivariateNormal(next_mu, next_cov))
        #
        # # linear weight lstm输入第i时刻隐变量，得到(hn,cn)，用于下一轮循环计算weight
        # _, weight_hidden_state = self.linears_weight_lstm(state.unsqueeze(dim=0), weight_hidden_state)
        #
        # state_sampled.append(state)





