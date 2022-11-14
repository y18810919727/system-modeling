#!/usr/bin/python
# -*- coding:utf8 -*-
import torchcde

from torchcde import LinearInterpolation
from model.ct_model.interpolation.cubic_intertpolation import CubicInterpolation


class MyZeroInterpolation(LinearInterpolation):
    def evaluate(self, t):
        fractional_part, index = self._interpret_t(t)
        coeffs = self._coeffs[..., index, :]
        return coeffs


class ZeroInterpolation(CubicInterpolation):

    def make_coefficients(self, ts, x):
        return torchcde.linear_interpolation_coeffs(x, t=ts)

    def make_interpolation(self, ts, coeffs):
        return MyZeroInterpolation(coeffs, t=ts)
