#!/usr/bin/python
# -*- coding:utf8 -*-

from torch import nn
from model.ct_model import ODEFunc
from model.ct_model.diffeq_solver import DiffeqSolver
from model.ct_model.ct_common import *


class ODERNN_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, ode_type='orth'):
        super(ODERNN_encoder, self).__init__()
        self.rnn_cell = torch.nn.GRUCell(input_dim, hidden_dim)

        self.gradient_net = MLP(hidden_dim, 2*hidden_dim, hidden_dim, num_mlp_layers=1)
        ode_func = ODEFunc(
            ode_net=self.gradient_net,
            inputs=None,
            ode_type=ode_type
        )
        self.diffeq_solver = DiffeqSolver(hidden_dim, ode_func, 'rk4',
                                          odeint_rtol=1e-6, odeint_atol=1e-7)

    def forward(self, seq, dts, state=None, last_state=False):
        states = []
        for x, dt in zip(seq, dts):
            state = self.rnn_cell(x, state)
            states.append(state)
            state = self.diffeq_solver(state, torch.stack([torch.zeros_like(dt), dt]))[-1]

        if last_state:
            return torch.stack(states, dim=0), state
        else:
            return torch.stack(states, dim=0)

