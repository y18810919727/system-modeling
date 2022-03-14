#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
from ct_model.ode_func import ODEFunc
from ct_model.diffeq_solver import DiffeqSolver
from ct_model.ct_common import linspace_vector
from model.common import MLP


class ODE_RNN(nn.Module):

    def __init__(self,
                 input_dim,
                 latent_dim,
                 ode_hidden_dim=50,
                 ode_solver='euler'
                 ):

        self.latent_dim, self.input_dim, self.ode_hidden_dim, self.ode_solver = \
            latent_dim, input_dim, ode_hidden_dim, ode_solver

        self.gru_cell = nn.GRUCell(
            input_dim,
            latent_dim * 2,
        )
        ode_func = ODEFunc(
            input_dim=latent_dim,
            ode_func_net=MLP(latent_dim, ode_hidden_dim, latent_dim, num_mlp_layers=2)
        )
        self.diffeq_solver = DiffeqSolver(latent_dim, ode_func, ode_solver,
                                          odeint_rtol=1e-3, odeint_atol=1e-4)

    def forward(self, data, h=None):
        data, tp = data[..., :-1], data[..., -1]
        assert data.size(-1) == self.input_dim

        l, bs, _ = data.shape
        device = data.device

        time_steps = torch.cumsum(
            torch.cat([
                torch.zeros_like(tp[0]),
                tp
            ], dim=0), dim=0)

        if h is None:
            h = self.init_hidden(bs, device)
        assert h.size(-1)==(self.latent_dim * 2)

        ode_state = h[0, ..., :self.latent_dim]
        gru_state = h[0, ..., -self.latent_dim:]

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 200
        ode_states_pre = []
        ode_states_post = []

        for i in range(data.size(0)):
            prev_t, t_i = time_steps[i], time_steps[i+1]

            if t_i - prev_t < minimum_step:
                inc = self.diffeq_solver.ode_func(prev_t, ode_state) * (t_i - prev_t)

                ode_sol = ode_state + inc
                ode_sol = torch.stack([ode_state, ode_sol], dim=0)

            else:
                n_intermediate_tp = max(2, ((t_i-prev_t)/minimum_step).int())
                time_points = linspace_vector(prev_t, t_i, n_intermediate_tp)
                ode_sol = self.diffeq_solver(ode_state, time_points)

            ode_state_pre = ode_sol[-1]
            xi = data[i]

            ode_states_pre.append(ode_state_pre)

            new_h = self.gru_cell(xi, torch.cat([ode_state_pre, gru_state], dim=-1))
            ode_state = new_h[..., :self.latent_dim]
            gru_state = new_h[..., :-self.latent_dim]

            ode_states_post.append(ode_state)

        ode_states_post = torch.stack(ode_states_post, dim=0)
        ode_states_pre = torch.stack(ode_states_pre, dim=0)

        # return ode_states_post, new_state, ode_state
        return ode_states_post, new_h

    def init_hidden(self, batch_size, device):
        """
        The num_layer of hidden state in GRU-Cell is 1
        Args:
            batch_size:
            device:

        Returns: num_layers, batch_size, 2 * latent_dim

        """
        return torch.zeros((1, batch_size, 2 * self.latent_dim)).to(device)
















