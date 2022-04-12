#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
import torch

import torch
from torch import nn


class ScaleNet(nn.Module):
    def __init__(self, scale, func):
        super(ScaleNet, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError('func must be a nn.Module')
        self.scale = scale.detach()
        self.func = func

    def __call__(self, t, z):
        gradient = self.func(t, z)
        return gradient * self.scale


def linspace_vector(start, end, n_points, device):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert(start.size() == end.size())
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(float(start), float(end), n_points, device=device)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res,
                             torch.linspace(float(start[i]), float(end[i]), n_points, device=device)), 0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res


def odeint_uniform_split(f, y0, tps, rtol=1e-7, atol=1e-9, method=None, options=None):
    batch_size, _ = y0.shape
    sols = []
    for b in range(batch_size):
        ys = odeint(f, y0[b:b+1], tps[:, b], rtol=rtol, atol=atol, method=method, options=options)
        sols.append(ys)
    return torch.cat(sols, dim=1)


def odeint_uniform_union(f, y0, tps, rtol=1e-7, atol=1e-9, method=None, options=None):
    T, N = tps.shape
    ts_norm = tps - tps[0:1]
    ts_ode = ts_norm[1:].reshape(-1).unique()
    ts_ode, _ = torch.cat([torch.zeros(1, device=ts_ode.device), ts_ode]).sort()
    ts_ode = torch.cat([ts_ode, ts_ode[-1:]+1e-3])
    Tidx = torch.zeros_like(ts_norm, dtype=torch.int64)
    for i in range(ts_ode.size(0)):
        Tidx[ts_norm == ts_ode[i]] = i
    ode_sols = odeint(f, y0, t=ts_ode, rtol=rtol, atol=atol, method=method, options=options)
    # for n in range(N):
    #     for t in range(T):
    #         assert ts_norm[t, n].float() == ts_ode[Tidx[n][t]]
    sols_uniform = torch.gather(ode_sols, 0, Tidx.unsqueeze(dim=-1).repeat(1, 1, y0.size(-1)))
    return sols_uniform


def odeint_scale(f, y0, tps, rtol=1e-7, atol=1e-9, method=None, options=None):

    if len(tps.shape) == 2:
        tps = tps.unsqueeze(dim=-1)
    sols = [y0]
    y = y0
    for i in range(tps.size(0)-1):
        dt = tps[i+1] - tps[i]
        nf = ScaleNet(scale=dt, func=f)
        ys = odeint(nf, y, torch.Tensor([0.0, 1.0]).to(y0.device), rtol=rtol, atol=atol, method=method, options=options)
        sols.append(ys[-1])

    return torch.stack(sols)

def odeint_uniform(f, y0, tps, rtol=1e-7, atol=1e-9, method=None, options=None):
    if len(tps.shape) == 2:
        tps = tps.unsqueeze(dim=-1)
    sols = [y0]
    y = y0
    for i in range(tps.size(0)-1):
        y = RK(tps[i], y, f, tps[i+1] - tps[i], scheme=method)
        sols.append(y)

    return torch.stack(sols)


def RK(t0, y, f, dt, scheme):
    # etplicit Runge Kutta methods
    # scheme in ['Euler', 'Midpoint', 'Kutta3', 'RK4']
    # t0 = t(t_n); optional t_half = t(t + 0.5 * dt), t_full = t(t + dt);
    # if not present, t0 is used (e.g. for piecewise constant inputs).

    t_half = t0 + 0.5*dt
    t_full = t0 + dt
    if scheme == 'euler':
        incr = dt * f(t0, y)
    elif scheme == 'midpoint':
        t1 = t0 if t_half is None else t_half
        k1 = f(t0, y)
        k2 = f(t1, y + dt * (0.5 * k1))  # t(t_n + 0.5 * dt)
        incr = dt * k2
    elif scheme == 'kutta3':
        t1 = t0 if t_half is None else t_half
        t2 = t0 if t_full is None else t_full
        k1 = f(t0, y)
        k2 = f(t1, y + dt * (0.5 * k1))  # t(t_n + 0.5 * dt)
        k3 = f(t2, y + dt * (- k1 + 2 * k2))  # t(t_n + 1.0 * dt)
        incr = dt * (k1 + 4 * k2 + k3) / 6
    elif scheme == 'rk4':
        t1 = t0 if t_half is None else t_half
        t2 = t0 if t_half is None else t_half
        t3 = t0 if t_full is None else t_full
        k1 = f(t0, y)
        k2 = f(t1, y + dt * (0.5 * k1))  # t(t_n + 0.5 * dt)
        k3 = f(t2, y + dt * (0.5 * k2))  # t(t_n + 0.5 * dt)
        k4 = f(t3, y + dt * k3)  # t(t_n + 1.0 * dt)
        incr = dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    else:
        return RK(t0, y, f, dt, 'rk4')

    return y + incr


def time_steps_increasing(time_steps, eps=1e-6):
    """
    To guarantee the time_steps be strictly increasing
    Args:
        time_steps:
    Returns:

    """
    with torch.no_grad():
        new_tps = torch.zeros_like(time_steps)
        base = torch.zeros_like(time_steps[0])
        for i in range(1, time_steps.size(0)):
            base[time_steps[i] <= time_steps[i-1]] += eps
            new_tps[i] = base + time_steps[i]
    return new_tps


def vector_sta(x, dx_dt):
    return torch.tanh(dx_dt) - x

def vector_normal(x, dx_dt):
    return dx_dt

def vector_orth(x, dx_dt):

    try:
        x_detach = x.detach()
        #.unsqueeze(dim=-1)
        x_norm = torch.linalg.norm(x_detach, dim=-1, keepdim=True)
        dx_dt_norm = torch.linalg.norm(dx_dt, dim=-1, keepdim=True)

        dot_sum = torch.sum(dx_dt*x_detach, dim=-1, keepdim=True)
        norm_dot = x_norm * dx_dt_norm
        cos_theta = dot_sum / norm_dot
        dx_dt_projection = x_detach / x_norm * (dx_dt_norm * cos_theta)

        orth_dx_dt = dx_dt - dx_dt_projection
        assert torch.cosine_similarity(orth_dx_dt, x_detach).mean() < 1e-5
    except AssertionError as e:
        print('orth: ', orth_dx_dt)
        print('x: ', x)
        np.save('dxdt.npy', orth_dx_dt.cpu().numpy())
        np.save('xt.npy', x_detach.cpu().numpy())

        raise e
    return orth_dx_dt
