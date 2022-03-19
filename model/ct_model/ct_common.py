#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
from torchdiffeq import odeint

import torch


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
