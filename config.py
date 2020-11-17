#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch



import argparse
parser = argparse.ArgumentParser(description='SE-VAE')

parser.add_argument('--dataset', type=str, help='fake or west', default='fake')
parser.add_argument('--net_type', type=str, default='lstm')
parser.add_argument('--random_seed', type=int,  default=None)
parser.add_argument('--input_size', type=int,  default=None)
parser.add_argument('--observation_size', type=int,  default=None)
parser.add_argument('--state_size', type=int,  default=8)
parser.add_argument('--k_size', type=int,  default=32)
parser.add_argument('--batch_size', type=int,  default=32)
parser.add_argument('--epochs', type=int,  default=800)
parser.add_argument('--eval_epochs', type=int,  default=10)
parser.add_argument('--version', type=str,  default='1')
parser.add_argument('--L', type=int,  default=5, help='The time for sampling VAE')
parser.add_argument('--num_layers', type=int,  default=1)
parser.add_argument('--lr', type=float,  default=1e-4)
parser.add_argument('--num_workers', type=int,  default=8)
parser.add_argument('--plt_cnt', type=int,  default=20)
parser.add_argument('--use_cuda', action='store_true', default=False)
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--test_id', type=str, default=None)
parser.add_argument('--reset', action='store_true', default=False, help='clear old logs and ckpt if save_dir exists')
parser.add_argument('--length', type=int,  default=100, help='The length of each episode')
parser.add_argument('--D', type=int,  default=30, help='The maximum distance for multi-step prediction')
parser.add_argument('--num_linears', type=int,  default=8, help='The number of combined linear models')
parser.add_argument('--dataset_window', type=int,  default=5, help='The size of moving window for generation '
                                                                     'episodes from dataset')
parser.add_argument('--lr_schedule', type=str, default=None)
parser.add_argument('--max_epochs_stop', type=int,  default=80, help='stop when the likelihood does not increase')




args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False


args.use_cuda = args.use_cuda if torch.cuda.is_available() else False

dirname_list = [args.dataset, 'v{}'.format(args.version)]
if args.save_dir != '':
    dirname_list.append(args.save_dir)
args.save_dir = '_'.join(dirname_list)

if args.dataset == 'fake':
    args.input_size = 3
    args.observation_size = 1
    args.state_size = 1

if args.dataset == 'west' or args.dataset == 'west-part':
    args.input_size = 4
    args.observation_size = 1



parser.add_argument('--num_linears', type=int,  default=8, help='The number of combined linear models')
parser.add_argument('--dataset_window', type=int,  default=5, help='The size of moving window for generation '
                                                                   'episodes from dataset')
parser.add_argument('--lr_schedule', type=str, default=None)
