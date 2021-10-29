#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch



class TestController:
    def __init__(self, set_value, input_dim, output_dim, length, default_action=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.length = length
        if not default_action:
            default_action = 0
        self.default_action = torch.FloatTensor([default_action] * input_dim)

    def solve(self, model, memory_state, last_seq_distribution=None):
        device = next(model.parameters()).device
        return None, self.default_action.to(device)




