#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
data_url = {
   'res_2018.0717-0723.csv': 'https://west-data.oss-cn-beijing.aliyuncs.com/res_2018.0717-0723.csv',
   'res_2019.0425-0503.csv': 'https://west-data.oss-cn-beijing.aliyuncs.com/res_2019.0425-0503.csv',
   'res_2019.1003-1011.csv': 'https://west-data.oss-cn-beijing.aliyuncs.com/res_2019.1003-1011.csv',
}

class SimpleLogger(object):
    def __init__(self, f, header='#logger output'):
        dir = os.path.dirname(f)
        self.dir = dir
        #print('test dir', dir, 'from', f)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(f, 'w') as fID:
            fID.write('%s\n'%header)
        self.f = f

    def __call__(self, *args):
        #standard output
        print(*args)
        #log to file
        try:
            with open(self.f, 'a') as fID:
                fID.write(' '.join(str(a) for a in args)+'\n')
        except:
            print('Warning: could not log to', self.f)


def normalize_seq(x, dim=0):
    import torch
    eps = 1e-12
    res = (x-torch.mean(x, dim=dim))/torch.sqrt(torch.var(x, dim=dim)).clamp_min(eps)
    # assert torch.mean(torch.mean(res,dim=dim)).norm() <1e-4
    # assert (torch.mean(torch.var(res,dim=dim))-1).norm() <1e-4
    return res


def cal_pearsonr(tensor_seq1, tensor_seq2):
    seq1 = tensor_seq1.detach().cpu().numpy().squeeze()
    seq2 = tensor_seq2.detach().cpu().numpy().squeeze()
    from scipy.stats import pearsonr
    return pearsonr(seq1, seq2)


def softplus(x, threshold=20):
    return torch.where(
        x < threshold, torch.log(1+torch.exp(x)), x
    )


def inverse_softplus(x, threshold=20):
    return torch.where(
        x < threshold, torch.log(torch.exp(x) - torch.ones_like(x)), x
    )

def detect_and_download(datapath):
    if os.path.exists(os.path.join('data', datapath)):
        return
    import urllib
    try:
        urllib.request.urlretrieve(data_url[datapath], filename=os.path.join('data', datapath))
    except Exception as e:
        print("Error occurred when downloading dataset file, error message:")
        print(e)


