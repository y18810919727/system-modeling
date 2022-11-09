#!/usr/bin/python
# -*- coding:utf8 -*-


class ConstInterpolation:
    def __init__(self, input):
        self._input = input

    def __call__(self, *args, **kwargs):
        return self._input