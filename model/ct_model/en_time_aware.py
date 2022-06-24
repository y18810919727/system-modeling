#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
from einops import rearrange,repeat

from model.common import DiagMultivariateNormal as MultivariateNormal
from common import logsigma2cov
import torch.nn.functional as F


class TimeAwareRNN(nn.Module):

    def __init__(self, k_in, k_out, k_state, dropout=0., interpol="constant", **kwargs):
        # potential kwargs: meandt, train_scheme, eval_scheme

        super().__init__()
        k_in -= 1
        self.k_in = k_in
        self.k_out = k_out
        self.k_state = k_state
        self.dropout = dropout
        self.criterion = nn.MSELoss()
        #self.criterion = nn.SmoothL1Loss()  # huber loss
        self.interpol = interpol

        self.pred_cell = HOGRUCell(k_in, k_out, k_state, scheme='RK4', dropout=dropout, **kwargs)
        self.en_cell = HOGRUCell(k_in+k_out, k_out, k_state, scheme='RK4', dropout=dropout, **kwargs)
        self.state0 = nn.Parameter(torch.zeros(k_state,), requires_grad=True)  # trainable initial state

    def forward_ori(self, cell, inputs, state0=None, dt=None, **kwargs):  # potential kwargs: dt
        """
        inputs size (batch_size, seq_len, k_in)
        state0 is None (self.state0 used) or initial state with shape (batch_size, k_state)
        """
        state = self.state0.unsqueeze(0).expand(inputs.size(0), -1) if state0 is None else state0
        outputs = []
        states = []

        # interpolation of inputs for higher order models (quick and dirty rather than very general)
        if self.interpol == "constant":
            inputs_half = inputs
            inputs_full = inputs
        elif self.interpol == "linear":
            # TODO: extrapolation?  Just for comparison.
            inputs_last = inputs[:, -1, :].unsqueeze(1)
            inputs_half = 0.5 * (inputs[:, :-1, :] + inputs[:, 1:, :])
            inputs_half = torch.cat([inputs_half, inputs_last], dim=1)  # constant approximation at end of seq.
            inputs_full = inputs[:, 1:, :]
            inputs_full = torch.cat([inputs_full, inputs_last], dim=1)  # constant approximation at end of seq.
        elif self.interpol == 'predicted':
            raise NotImplementedError('Interpolation %s not yet implemented.' % str(self.interpol))
        else:
            raise NotImplementedError('invalid interpolation' % str(self.interpol))

        # forward pass through inputs sequence
        for i in range(inputs.size(1)):
            x0 = inputs[:, i, :]
            x_half = cell.expand_input(inputs_half[:, i, :])
            x_full = cell.expand_input(inputs_full[:, i, :])
            dt_i = None if dt is None else dt[:, i, :]
            output, state = cell(x0, state, dt=dt_i, x_half=x_half, x_full=x_full)  # output (batch, 2)
            outputs.append(output)
            states.append(state)

        outputs = torch.stack(outputs, dim=1)  # outputs: (batch, seq_len, 2)
        states = torch.stack(states, dim=1)

        return outputs, states

    def forward_prediction(self, external_input_seq, n_traj=16, max_prob=False, memory_state=None):

        l, batch_size, _ = external_input_seq.size()
        predicted_seq, states, memory_state= self.forward_sequence(
            self.pred_cell, external_input_seq, memory_state
        )

        predicted_seq_sample = repeat(predicted_seq, 'l b c -> l b n c', n=n_traj)
        predicted_dist = MultivariateNormal(
            predicted_seq, torch.diag_embed(torch.zeros_like(predicted_seq))
        )
        outputs = {
            'predicted_seq_sample': predicted_seq_sample,
            'predicted_dist': predicted_dist,
            'predicted_seq': predicted_seq
        }
        return outputs, memory_state

    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
        """
        训练时：估计隐变量后验分布，并采样，用于后续计算模型loss
        测试时: 为后续执行forward_prediction计算memory_state(h, rnn_hidden)
        Args:
            external_input_seq: 系统输入序列(进出料浓度、流量) (len, batch_size, input_size)
            observations_seq: 观测序列(泥层压强) (len, batch_size, observations_size)
            memory_state: 模型支持长序列预测，之前forward过程产生的记忆信息压缩在memory_state中

        Returns:

        """

        l, batch_size, _ = external_input_seq.size()

        predicted_seq, states, memory_state = self.forward_sequence(
            self.en_cell, torch.cat([observations_seq, external_input_seq], dim=-1),  memory_state
        )

        outputs = {
            'state': states,
            'predicted_seq': predicted_seq,
        }
        return outputs, memory_state

    def call_loss(self, external_input_seq, observations_seq, memory_state=None):

        en_length = external_input_seq.size(0)//2

        historical_input = external_input_seq[:-en_length]
        historical_ob = observations_seq[:-en_length]
        future_input = external_input_seq[-en_length:]

        en_outputs, memory_state = self.forward_posterior(historical_input, historical_ob, memory_state)
        pred_outputs, memory_state = self.forward_prediction(future_input, n_traj=1, memory_state=memory_state)

        en_seq_plus_pred_seq = torch.cat([en_outputs['predicted_seq'], pred_outputs['predicted_seq']], dim=0)

        return {
            'loss': F.mse_loss(en_seq_plus_pred_seq, observations_seq),
            'kl_loss': 0,
            'likelihood_loss': 0
        }

    def forward_sequence(self, cell, seq, memory_state):

        seq, dt = seq[..., :-1], seq[..., -1:]
        batch_size = seq.size(1)
        state = self.state0.repeat(batch_size, 1) if memory_state is None else memory_state['state']
        last_dt = 0 if memory_state is None else memory_state['last_dt']

        ori_dt = dt
        dt = torch.cat([torch.zeros_like(dt[0:1]), dt[:-1]], dim=0)
        dt[0] = last_dt

        outputs, states = self.forward_ori(
            cell,
            inputs=rearrange(seq, 'l b c -> b l c'),
            state0=state,
            dt=rearrange(dt, 'l b 1 -> b l 1')
        )

        new_memory_state = {
            'state': states[:, -1, :],
            'last_dt': ori_dt[-1]
        }

        outputs = torch.cat(
            [
                cell.Ly(state).unsqueeze(dim=1),
                outputs[:, :-1, :]
            ],
            dim=1
        )
        states = torch.cat(
            [
                state.unsqueeze(dim=1),
                states[:, :-1, :]
            ],
            dim=1
        )
        predicted_seq, states = [rearrange(x, 'b l c -> l b c') for x in [outputs, states]]

        return predicted_seq, states, new_memory_state


    def decode_observation(self, outputs, mode='sample'):
        """

        Args:
            state: with shape (len, batch_size, state_size)
            mode: dist or sample

        Returns:

        """
        mean = outputs['predicted_seq']
        cov = torch.diag_embed(torch.zeros_like(mean))
        observations_normal_dist = MultivariateNormal(
            mean, cov
        )
        if mode == 'dist':
            return observations_normal_dist
        elif mode == 'sample':
            return mean


class HOGRUCell(nn.Module):
    """
    higher order GRU cell; 1st order with equidistant samples is equivalent to standard GRU cell
    """
    def __init__(self, k_in, k_out, k_state, dropout=0., meandt=1, scheme='RK4', **kwargs):

        super().__init__()
        self.k_in = k_in
        self.k_out = k_out
        self.k_state = k_state
        self.meandt = meandt
        self.state_size = k_state
        self.scheme = scheme

        self.expand_input = nn.Sequential(nn.Linear(k_in, k_state), nn.Tanh())

        self.Lx = nn.Linear(k_state, 3 * k_state, bias=False)
        self.Lh_gate = nn.Linear(k_state, 2 * k_state, bias=True)
        self.Lh_lin = nn.Linear(k_state, k_state, bias=True)

        self.Ly = nn.Sequential(
            nn.Linear(k_state, k_state, bias=False),
            nn.Tanh(),
            nn.Linear(k_state, k_out, bias=False)
        )
        self.dropout = nn.Dropout(dropout)

        self.init_params()
        self.train()

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.k_state)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

        #initialize GRU reset gate bias to -1
        torch.nn.init.constant_(self.Lh_gate.bias[self.k_state:], -1)

    def forward(self, x, state, dt, x_half=None, x_full=None, **kwargs):
        scheme = self.scheme

        state_new = self.dropout(RK(self.expand_input(x), state, self.f, dt, scheme, x_half=x_half, x_full=x_full))  # (batch, k_gru)
        #TODO check if this is best way to include input expansion
        y_new = self.Ly(state_new)
        return y_new, state_new

    def f(self, x, h):
        # x (batch, k_in), h (batch, k_gru)
        Lx = self.Lx(x)
        Lh_gate = self.Lh_gate(h)
        Lx_gate, Lx_lin = torch.split(Lx, [2 * self.k_state, self.k_state], dim=1)
        gates = torch.sigmoid(Lx_gate + Lh_gate)  # (batch, 2 * k_gru)
        z, r = torch.split(gates, [self.k_state, self.k_state], dim=1)

        return z * (torch.tanh(Lx_lin + self.Lh_lin(r * h)) - h) / self.meandt


def RK(x0, y, f, dt, scheme, x_half=None, x_full=None):
    # explicit Runge Kutta methods
    # scheme in ['Euler', 'Midpoint', 'Kutta3', 'RK4']
    # x0 = x(t_n); optional x_half = x(t + 0.5 * dt), x_full = x(t + dt);
    # if not present, x0 is used (e.g. for piecewise constant inputs).

    if scheme == 'Euler':
        incr = dt * f(x0, y)
    elif scheme == 'Midpoint':
        x1 = x0 if x_half is None else x_half
        k1 = f(x0, y)
        k2 = f(x1, y + dt * (0.5 * k1))  # x(t_n + 0.5 * dt)
        incr = dt * k2
    elif scheme == 'Kutta3':
        x1 = x0 if x_half is None else x_half
        x2 = x0 if x_full is None else x_full
        k1 = f(x0, y)
        k2 = f(x1, y + dt * (0.5 * k1))  # x(t_n + 0.5 * dt)
        k3 = f(x2, y + dt * (- k1 + 2 * k2))  # x(t_n + 1.0 * dt)
        incr = dt * (k1 + 4 * k2 + k3) / 6
    elif scheme == 'RK4':
        x1 = x0 if x_half is None else x_half
        x2 = x0 if x_half is None else x_half
        x3 = x0 if x_full is None else x_full
        k1 = f(x0, y)
        k2 = f(x1, y + dt * (0.5 * k1))  # x(t_n + 0.5 * dt)
        k3 = f(x2, y + dt * (0.5 * k2))  # x(t_n + 0.5 * dt)
        k4 = f(x3, y + dt * k3)  # x(t_n + 1.0 * dt)
        incr = dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    elif scheme == 'ode':
        from torchdiffeq import odeint_adjoint as odeint

        class AugF(nn.Module):
            def __init__(self, f, x0, x_full, dt):
                super(AugF, self).__init__()
                self.x0 = x0
                self.x_full = x_full
                self.f = f
                self.dt = dt
                self.cal_times = 0

            def forward(self, t, yt):
                self.cal_times += 1
                return self.dt*f(x0 + (x_full - x0) * t, yt)
        # import pdb
        # pdb.set_trace()

        #return odeint(AugF(f, x0, x_full, dt), y, torch.linspace(0, 1, 2).to(dt.device), rtol=1e-2, atol=1e-3)[-1]
        """
        SOLVERS = {
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        }
        """
        aug_f = AugF(f, x0, x_full, dt)
        result = odeint(aug_f, y, torch.linspace(0, 1, 2).to(dt.device), method='tsit5')[-1]
        return result
    else:
        raise NotImplementedError

    return y + incr

