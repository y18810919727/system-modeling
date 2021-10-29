#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch

# 归一化
class Scale:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.input_nums = self.mean.shape[0]

    def scale_array(self, array):
        """

        Args:
            array: (..., input_nums)

        Returns:

        """
        assert array.shape[-1] == self.input_nums
        return (array - self.mean) / self.std

    def scale_scalar(self, array, pos):
        """

        Args:
            pos: 需要归一化的对象在mean 和 std中对应的位置

        Returns:

        """

        return (array - self.mean[..., pos]) / self.std[..., pos]

    def unscale_array(self, array):
        """

        Args:
            array: (..., input_nums)

        Returns:

        """
        assert array.shape[-1] == self.input_nums
        return array * self.std + self.mean

    def unscale_scalar(self, array, pos):
        """

        Args:
            array: (..., input_nums)
            pos: 需要反归一化的对象在mean和std中对应的位置

        Returns:

        """

        return array * self.std[..., pos] + self.mean[..., pos]
