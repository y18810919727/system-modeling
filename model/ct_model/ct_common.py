#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

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
