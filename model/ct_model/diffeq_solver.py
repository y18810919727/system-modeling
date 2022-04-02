#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
from torchdiffeq import odeint as odeint
from model.ct_model.ct_common import odeint_uniform, time_steps_increasing, odeint_uniform_union, odeint_scale, odeint_uniform_split
from lib.util import TimeRecorder
import sys
ODE_TIME_EVAL = False



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
        #
        elif (time_steps_to_predict[:, 1:] - time_steps_to_predict[:, :-1] == 0).all():
            time_steps_to_predict = time_steps_to_predict[:, 0]
            return odeint(self.ode_func, first_point, time_steps_to_predict,
                          rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)

        else:
            # if self.ode_method == 'dopri5':
            #     return odeint_uniform_union(self.ode_func, first_point, time_steps_to_predict,
            #                           rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
            # else:
            #     return odeint_uniform(self.ode_func, first_point, time_steps_to_predict,
            #                       rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)

            tr = TimeRecorder()
            with tr('scale'):
                res_scale = odeint_scale(self.ode_func, first_point, time_steps_to_predict,
                                         rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)

            if not ODE_TIME_EVAL:
                return res_scale
            else:

                tol = [
                    ((1e-1, 1e-2), '12'),
                    ((1e-2, 1e-3), '23'),
                    ((1e-4, 1e-5), '45'),
                    ((1e-6, 1e-7), '67'),
                    ((1e-8, 1e-9), '89'),
                ]

                ans = []
                for (rtol, atol), name in tol:
                    with tr(name):
                        ans.append(odeint_uniform_union(self.ode_func, first_point, time_steps_to_predict, rtol=rtol, atol=atol, method=self.ode_method))

                with tr('split'):
                    res_split = odeint_uniform_split(self.ode_func, first_point, time_steps_to_predict,
                                           rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)

                with tr('rk'):
                    res_rk = odeint_uniform(self.ode_func, first_point, time_steps_to_predict,
                                          rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)

                with tr('scale'):
                    res_scale = odeint_scale(self.ode_func, first_point, time_steps_to_predict,
                                           rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
                for i in range(len(ans)):
                    l2 = torch.dist(ans[i], res_split)
                    cos = torch.cosine_similarity(ans[i], res_split, dim=-1).mean()
                    print('Name: {}, Time: {}, l2: {}, cos: {}'.format(tol[i][1], tr.infos[tol[i][1]], l2, cos))

                l2 = torch.dist(res_rk, res_split)
                cos = torch.cosine_similarity(res_rk, res_split, dim=-1).mean()
                # print('rk', tr.infos['rk'], torch.dist(res_rk, res_split))
                print('Name: {}, Time: {}, l2: {}, cos: {}'.format('rk', tr.infos['rk'], l2, cos))

                l2 = torch.dist(res_scale, res_split)
                cos = torch.cosine_similarity(res_scale, res_split, dim=-1).mean()
                # print('rk', tr.infos['rk'], torch.dist(res_rk, res_split))
                print('Name: {}, Time: {}, l2: {}, cos: {}'.format('scale', tr.infos['scale'], l2, cos))

                return res_split








