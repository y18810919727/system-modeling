#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch



import argparse
parser = argparse.ArgumentParser(description='SE-VAE')

parser.add_argument('--dataset', type=str, help='fake or real', default='fake')
parser.add_argument('--net_type', type=str, default='lstm')
parser.add_argument('--random_seed', type=int,  default=None)
parser.add_argument('--input_size', type=int,  default=None)
parser.add_argument('--observation_size', type=int,  default=None)
parser.add_argument('--state_size', type=int,  default=None)
parser.add_argument('--k_size', type=int,  default=32)
parser.add_argument('--batch_size', type=int,  default=32)
parser.add_argument('--epochs', type=int,  default=600)
parser.add_argument('--eval_epochs', type=int,  default=20)
parser.add_argument('--version', type=int,  default=1)
parser.add_argument('--L', type=int,  default=5, help='The time for sampling VAE')
parser.add_argument('--num_layers', type=int,  default=1)
parser.add_argument('--lr', type=float,  default=1e-3)
parser.add_argument('--num_workers', type=int,  default=8)
parser.add_argument('--plt_cnt', type=int,  default=20)
parser.add_argument('--use_cuda', action='store_true', default=False)
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--test_id', type=str, default=None)
parser.add_argument('--reset', action='store_true', default=False, help='clear old logs and ckpt if save_dir exists')



args = parser.parse_args()
if args.dataset == 'fake':
    args.input_size = 3
    args.observation_size = 1
    args.state_size = 1

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

args.use_cuda = args.use_cuda if torch.cuda.is_available() else False

dirname_list = []
if args.save_dir != '':
    dirname_list.append(args.save_dir)
dirname_list.append('v{}'.format(args.version))
args.save_dir = '_'.join(dirname_list)




