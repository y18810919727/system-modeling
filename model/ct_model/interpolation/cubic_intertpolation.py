#!/usr/bin/python
# -*- coding:utf8 -*-
import torchcde

class CubicInterpolation:
    def __init__(self, ts, x):
        """
        Args:
            ts: len, batch_size, 1
            x:  len, batch_size, dim
        """
        length, batch_size, _ = x.shape
        if ts.shape == (length, batch_size, 1):
            ts = ts[:, 0, 0]
        elif ts.shape == (length, 1):
            ts = ts[:, 0]
        elif ts.shape == (length,):
            pass
        ts = ts.contiguous()
        x = x.transpose(0, 1).contiguous()
        # ts = ts.transpose(0, 1)
        # x = torch.cat([ts, x], dim=-1)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x, ts)
        # coeffs = torchcde.natural_cubic_spline_coeffs(x, t=ts)
        self.X = torchcde.CubicSpline(coeffs, t=ts)

    def __call__(self, t):
        """

        Args:
            t: single tensor or 1d tensor with shape (1)
        Returns:

        """
        t = t.squeeze(dim=0) if t.ndim == 1 else t
        return self.X.evaluate(t)
