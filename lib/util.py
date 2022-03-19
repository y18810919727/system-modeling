#!/usr/bin/python
# -*- coding:utf8 -*-

import os
import yaml
import numpy as np
import json
# from ruamel_yaml import
import time

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


class TimeRecorder:
    def __init__(self):
        self.infos = {}

    def __call__(self, info, *args, **kwargs):
        class Context:
            def __init__(self, recoder, info):
                self.recoder = recoder
                self.begin_time = None
                self.info = info

            def __enter__(self):
                self.begin_time = time.time()

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.recoder.infos[self.info] = time.time() - self.begin_time

        return Context(self, info)

    def __str__(self):
        return ' '.join(['{}:{:.2f}s'.format(info, t) for info, t in self.infos.items()])

