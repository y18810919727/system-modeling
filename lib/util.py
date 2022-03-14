#!/usr/bin/python
# -*- coding:utf8 -*-

import os
import yaml
import numpy as np
import json
# from ruamel_yaml import

from omegaconf import DictConfig, OmegaConf


def load_DictConfig(path, name):
    file = os.path.join(
        path, name
    )
    if not os.path.exists(os.path.join(path, name)):
        return None

    config = OmegaConf.load(file)

    return DictConfig(config)


def write_DictConfig(path, name, exp_dict: DictConfig):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, name), "w", encoding="utf-8") as f:
        OmegaConf.save(exp_dict, f)


def load_yaml(path, name):

    file = os.path.join(
        path, name
    )
    if not os.path.exists(os.path.join(path, name)):
        return None

    with open(file, "r", encoding="utf-8") as f:
        config = yaml.load(f)

    return DictConfig(config)


