#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import torch
from torch import nn
from common import softplus, inverse_softplus, cov2logsigma, logsigma2cov
from model.func import weighted_linear, normal_differential_sample


class DBlock(nn.Module):
    """ A basie building block for parametralize a normal distribution.
    It is corresponding to the D operation in the reference Appendix.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = t * torch.sigmoid(self.fc2(input))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma


class PreProcess(nn.Module):

    def __init__(self, input_size, processed_x_size):
        super(PreProcess, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = self.fc2(t)
        return t
