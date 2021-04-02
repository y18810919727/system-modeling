#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch.distributions.normal import Normal


class CEMPlanning:

    def __init__(self, input_dim, output_dim, length, num_samples=32, max_iters=50, device='cpu'):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.length = length
        self.num_samples = num_samples
        self.device = device
        self.max_iters = max_iters


    def eval(self, state, action):
        """

        Args:
            new_state: with shape(samples, length, output_dim)
            action: with shape (samples, length, input_dim)

        Returns: cost shape with (samples, 1)
        """

        return None

    def solve(self, model, memory_state, last_seq_distribution=None):
        """

        Args:
            model:
            last_seq_distribution:
            memory_state:

        Returns:

            new_seq_distribution: An normal distribution with shape [length, ]


        """
        if last_seq_distribution is None:
            mean = torch.zeros((self.length, self.input_dim)).to(self.device)
            scale = torch.zeros((self.length, self.input_dim)).to(self.device)
        else:
            mean_last = torch.zeros((1, self.input_dim)).to(self.device)
            scale_last = torch.zeros((1, self.input_dim)).to(self.device)
            mean = torch.cat([last_seq_distribution.mean[:-1], mean_last], dim=0)
            scale = torch.cat([last_seq_distribution.scale[:-1], scale_last], dim=0)
        last_seq_distribution = Normal(mean, scale)

        new_seq_distribution = None
        for _ in range(self.max_iters):
            """
            1. 从last_seq_distribution采样
            2. 优化last_seq_distribution
            """
            pass

        # 测试返回数据用
        new_seq_distribution = Normal(
            torch.zeros((self.length, self.input_dim)).to(self.device),
            torch.ones((self.length, self.input_dim)).to(self.device)
        )
        return new_seq_distribution



