#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
import numpy as np
from einops import rearrange, repeat
def sq_dist3(X1, X2, ell=1.0):
    N = X1.shape[0]
    X1  = X1 / ell
    X1s = torch.sum(X1**2, dim=-1).view([N,-1,1])
    X2  = X2 / ell
    X2s = torch.sum(X2**2, dim=-1).view([N,1,-1])
    sq_dist = -2*X1@X2.transpose(-1,-2) + X1s + X2s
    return sq_dist

def Klinear(X1, X2, ell=1.0, sf=1.0, eps=1e-5):
    dnorm2 = sq_dist(X1,X2,ell) if X1.ndim==2 else sq_dist3(X1,X2,ell)
    K_ = sf**2 * torch.exp(-0.5*dnorm2)
    if X1.shape[-2]==X2.shape[-2]:
        return K_ + torch.eye(X1.shape[-2],device=X1.device)*eps
    return K_

def sq_dist(X1, X2, ell=1.0):
    """
    (X1-X2) @ (X1-X2).T

    :param X1:  n * n
    :param X2:  n * n
    :param ell:
    :return:  n * n
    """
    X1  = X1 / ell
    X1s = torch.sum(X1**2, dim=-1).view([-1,1])
    X2  = X2 / ell
    X2s = torch.sum(X2**2, dim=-1).view([1,-1])
    sq_dist = -2*torch.mm(X1,X2.t()) + X1s + X2s
    return sq_dist

def batch_sq_dist(x, y, ell=1.0):
    '''
    Modified from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
    Input: x is a bxNxd matrix y is an optional bxMxd matirx
    Output: dist is a bxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
    i.e. dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2
    '''
    assert x.ndim==3, 'Input1 must be 3D, not {x.shape}'
    y = y if y.ndim==3 else torch.stack([y]*x.shape[0])
    assert y.ndim==3, 'Input2 must be 3D, not {y.shape}'
    x,y = x/ell, y/ell
    x_norm = (x**2).sum(2).view(x.shape[0],x.shape[1],1)
    y_t = y.permute(0,2,1).contiguous()
    y_norm = (y**2).sum(2).view(y.shape[0],1,y.shape[1])
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    dist[dist != dist] = 0 # replace nan values with 0

    return torch.clamp(dist, 0.0, np.inf)

def K(X1, X2, ell=1.0, sf=1.0, eps=1e-5):
    dnorm2 = sq_dist(X1,X2,ell) if X1.ndim==2 else sq_dist3(X1,X2,ell)
    K_ = sf**2 * torch.exp(-0.5*dnorm2)
    if X1.shape[-2]==X2.shape[-2]:
        return K_ + torch.eye(X1.shape[-2],device=X1.device)*eps
    return K_

class KernelInterpolation:
    def __init__(self, X, y, eps=1e-5, kernel='exp'):
        """

        Args:
            X: time steps (len, batch size, 1)
            y: series (len, batch size, dim)
            eps:
            kernel:
        """
        X = X.transpose(0, 1)
        y = y.transpose(0, 1)
        batch_size, N, n_out = y.shape
        device = y.device
        self.batch_size, self.device = batch_size, device
        sf = 1.0 * torch.ones([batch_size, 1, 1], device=device, dtype=torch.float32)
        ell = 0.5 * torch.ones([batch_size, 1, 1], device=device, dtype=torch.float32)
        self.sf = sf
        self.ell = ell
        self.X = X
        self.y = y
        self.eps = eps
        self.K = K if kernel == 'exp' else Klinear
        self.KXX_inv_y = y.solve(self.K(X, X, ell, sf, eps))[
            0]  # A.solve(y) <=> the solution of Ax=y and LU factorization, the first element(0) is x.

    def __call__(self, x):
        """
        Interpolate the value at time step x
        Args:
            x: (L, B) or (L, ), L is the length of the time series, B is the batch size

        Returns:

        """
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x).to(self.device)
        if x.ndim == 0:
            # x = x.unsqueeze(-1)
            x = x.repeat(self.batch_size, 1, 1)
        elif x.ndim == 1:
            if x.shape[0] == self.batch_size:
                x = rearrange(x, 'b -> b o o', o=1)
            else:
                x = repeat(x, 'o -> b c o', b=self.batch_size, c=1)
                # >> > repeat(image, 'h w -> (h h2) (w w2)', h2=2, w2=2).shape
        elif x.ndim == 2:
            x = x.unsqueeze(dim=1) # (batch size, 1) -> (batch size, 1, 1)
        kxX = self.K(x, self.X, self.ell, self.sf, self.eps)
        out = kxX @ self.KXX_inv_y
        # (batch_size, len, n_out) -> (batch_size, len, n_out)
        return out[:, 0]

if __name__ == '__main__':
    X = torch.randn((3, 3, 3))
    t = torch.linspace(0, 1, 3).reshape((3, 1, 1)).repeat((1, 3, 1))
    print(X.shape, t.shape)
    device = X.device
    N = 3
    # sfs = 1.0 * torch.ones([N, 1, 1], device=device, dtype=torch.float32)
    # ells = 0.5 * torch.ones([N, 1, 1], device=device, dtype=torch.float32)

    inter = KernelInterpolation(t, X, eps=1e-5, kernel='exp')

    pos = torch.mean(t, dim=1)
    print(inter(pos).shape)

