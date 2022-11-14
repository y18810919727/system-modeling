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
        ts, x = self.process_input(ts, x)
        coeffs = self.make_coefficients(ts, x)
        self.X = self.make_interpolation(ts, coeffs)

    def process_input(self, ts, x):

        length, batch_size, _ = x.shape
        if ts.shape == (length, batch_size, 1):
            ts = ts[:, 0, 0]
        elif ts.shape == (length, 1):
            ts = ts[:, 0]
        elif ts.shape == (length,):
            pass
        ts = ts.contiguous()
        x = x.transpose(0, 1).contiguous()
        return ts, x

    def make_coefficients(self, ts, x):
        return torchcde.hermite_cubic_coefficients_with_backward_differences(x, ts)

    def make_interpolation(self, ts, coeffs):
        return torchcde.CubicSpline(coeffs, t=ts)

    def __call__(self, t):
        """

        Args:
            t: single tensor or 1d tensor with shape (1)
        Returns:

        """
        t = t.squeeze(dim=0) if t.ndim == 1 else t
        return self.X.evaluate(t)
