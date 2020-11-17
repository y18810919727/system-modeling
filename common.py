#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch

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
    """

    Args:
        tensor_seq1: shape (len, xxx)
        tensor_seq2: shape (len, xxx)

    Returns:

    """

    if len(tensor_seq1.shape) >= 2:
        pearson_list = [cal_pearsonr(tensor_seq1[:, i], tensor_seq2[:, i]) for i in range(tensor_seq1.shape[1])]
        return np.mean(pearson_list)
    from scipy.stats import pearsonr

    seq1 = tensor_seq1.detach().cpu().numpy()
    seq2 = tensor_seq2.detach().cpu().numpy()
    return pearsonr(seq1, seq2)[0]


def RRSE(y_pred, y_gt):
    assert y_gt.shape == y_pred.shape
    if len(y_gt.shape) == 3:
        return torch.mean(
            torch.stack(
                [RRSE(y_pred[:, i], y_gt[:, i]) for i in range(y_gt.shape[1])]
            )
        )

    elif len(y_gt.shape) == 2:
        # each shape (n_seq, n_outputs)
        se = torch.sum((y_gt - y_pred)**2, dim=0)
        rse = se / torch.sum(
            (y_gt - torch.mean(y_gt, dim=0))**2, dim=0
        )
        return torch.mean(torch.sqrt(rse))
    else:
        raise AttributeError


def softplus(x, threshold=20):
    return torch.where(
        x < threshold, torch.log(1+torch.exp(x)), x
    )


def inverse_softplus(x, threshold=20):
    return torch.where(
        x < threshold, torch.log(torch.exp(x) - torch.ones_like(x)), x
    )


def logsigma2cov(logsigma):
    return torch.diag_embed(softplus(logsigma)**2)


def cov2logsigma(cov):
    return inverse_softplus(torch.sqrt(cov.diagonal(dim1=-2, dim2=-1)))


def normal_interval(dist, e):
    return dist.loc - e * torch.sqrt(dist.covariance_matrix.diagonal(dim1=-2, dim2=-1)), \
           dist.loc + e * torch.sqrt(dist.covariance_matrix.diagonal(dim1=-2, dim2=-1))



def detect_download(data_urls, base):
    import pandas as pd
    data_paths = []

    def download(url, path):
        import urllib
        print("Downloading file %s from %s:" % (path, url))
        try:
            urllib.request.urlretrieve(url, filename=os.path.join( path))
            return path
        except Exception as e:
            print("Error occurred when downloading file %s from %s, error message :" % (path, url))
            return None
    for name, url in zip(data_urls['object'], data_urls['url']):
        path = os.path.join(base, name)
        if not os.path.exists(path) and not download(url, path):
            pass
        else:
            data_paths.append(path)
    return data_paths


def merge_first_two_dims(tensor):
    size = tensor.size()
    return tensor.contiguous().reshape(-1, *size[2:])


def split_first_dim(tensor, sizes=None):
    if sizes is None:
        sizes = tensor.size()[0]
    import numpy
    assert type(sizes) is tuple and numpy.prod(sizes) == tensor.size()[0]
    return tensor.contiguous().reshape(*sizes, *tensor.size()[1:])

