#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import json
import time

from omegaconf import DictConfig, OmegaConf, ListConfig

def my_JSON_serializable(d):
    if isinstance(d, torch.Tensor):
        return json.dumps(d.detach().cpu().numpy().tolist())
    elif isinstance(d, dict):
        nd = {}
        for k, v in d.items():
            nd[k] = my_JSON_serializable(v)
        return json.dumps(nd)
    else:
        return json.dumps(d)

def my_JSON_load(str):

    d = json.loads(str)

    if isinstance(d, list):
        return np.array(d, dtype=np.float32)
    elif isinstance(d, dict):
        nd = {}
        for k, v in d.items():
            nd[k] = my_JSON_load(v)
        return json.dumps(nd)
    else:
        return json.dumps(d)


def DictConfig2dict(config):
    if isinstance(config, DictConfig):
        dict_config =dict(config)
        return {key: DictConfig2dict(value) for key, value in dict_config.items()}
    elif isinstance(config, ListConfig):
        return [DictConfig2dict(x) for x in config]
    else:
        return config






class Timer(object):
    def __init__(self, info):
        self.begin_time = 0
        self.time_used = 0

    def __enter__(self):
        self.begin_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time_used = time.perf_counter() - self.begin_time


def dict_to_Tensor(memory_state, device=torch.device('cpu')):
    new_memory_state = {}
    for k, v in memory_state.items():
        new_memory_state[k] = torch.Tensor(
            np.array(v if isinstance(v, list) else json.loads(v), dtype=np.float32),
        ).to(device)
    return new_memory_state
