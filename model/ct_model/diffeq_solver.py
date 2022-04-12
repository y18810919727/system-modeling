#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
# from torchdiffeq import odeint as odein
from torchdiffeq import odeint_adjoint as odeint
from model.ct_model.ct_common import odeint_uniform, time_steps_increasing, odeint_uniform_union, odeint_scale, odeint_uniform_split
from lib.util import TimeRecorder
import sys

class DiffeqSolver(nn.Module):
    def __init__(self, input_dim, ode_func, method,
            odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu")):
        super(DiffeqSolver, self).__init__()
        self.ode_method = method
        self.device = device
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict):
        """
        # Decode the trajectory through ODE Solver
        """
        time_steps_to_predict = time_steps_increasing(time_steps_to_predict)
        if len(time_steps_to_predict.shape) == 1:
            return odeint(self.ode_func, first_point, time_steps_to_predict,
                          rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)

        elif (time_steps_to_predict[:, 1:] - time_steps_to_predict[:, :-1] == 0).all():
            time_steps_to_predict = time_steps_to_predict[:, 0]
            return odeint(self.ode_func, first_point, time_steps_to_predict,
                          rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)

        else:
            res_scale = odeint_scale(self.ode_func, first_point, time_steps_to_predict,
                                     rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
            return res_scale
