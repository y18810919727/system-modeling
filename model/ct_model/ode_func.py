#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
import common
from model.common import MLP
from model.ct_model.ct_common import vector_orth, vector_sta, vector_normal

class ODEFunc(nn.Module):
    def __init__(self, input_dim, ode_hidden_dim, ode_num_layers, ode_type='sta', device=torch.device("cpu")):

        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()

        self.input_dim = input_dim
        self.device = device

        self.gradient_net = MLP(input_dim, ode_hidden_dim, input_dim, num_mlp_layers=ode_num_layers)
        common.init_network_weights(self.gradient_net)
        # normal: mlp; sta: mlp(tanh) - ht; orth: mlp * sin
        if ode_type == 'normal':
            self.gradient_func = vector_normal
        elif ode_type == 'sta':
            self.gradient_func = vector_sta
        elif ode_type == 'orth':
            self.gradient_func = vector_orth
        else:
            raise NotImplementedError(f'{ode_type}is not implemented')



    def forward(self, t_local, y, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        dy_dt = self.gradient_net(y)
        return self.gradient_func(y, dy_dt)

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)
