#!/usr/bin/python
# -*- coding:utf8 -*-


from model.ct_model.interpolation.const_interpolation import ConstInterpolation
from model.ct_model.interpolation.kernel_interpolation import KernelInterpolation
from model.ct_model.interpolation.zero_interpolation import ZeroInterpolation
from model.ct_model.interpolation.cubic_intertpolation import CubicInterpolation

supported_batched_interpolation_types = ['const', 'gp']


def interpolate(inter_type, ts, x, batched=False):
    if batched and inter_type not in supported_batched_interpolation_types:
        raise '{} is not supported for Batched interpolation. Please choosing anyone in {}.'.format(
            inter_type, ','.join(supported_batched_interpolation_types)
        )
    if inter_type == 'gp':
        interpolation_class = KernelInterpolation
    elif inter_type == 'cubic':
        interpolation_class = CubicInterpolation
    elif inter_type == 'zero':
        interpolation_class = ZeroInterpolation
    else:
        raise NotImplementedError

    return interpolation_class(ts, x)
