#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import pandas
from matplotlib import pyplot as plt

import numpy as np

begin_state_mu = 80
begin_state_sigma = 2.0

def generate_fake_data(
        name,
        begin_state_mu=begin_state_mu,
        begin_state_sigma=begin_state_sigma,
        sigma=2.0,
        delta=6.0,
        pressure_bias=10,
        D=0.78,
        a=1.21,
        b=0.239,
        c=1.051,
        plot=False,
        transition_mu=None,
        transition_sigma=None,
        L=100,
        n=1000,
):
    """
    :param begin_state_mu: 初始干砂质量均值
    :param begin_state_sigma: 初始干砂质量的标准差
    :param sigma: 干砂状态转移过程噪音的标准差
    :param delta: 观测误差的标准差
    :param D: 观测系数
    :param a: 转移过程参数的二次项
    :param b: 转移过程参数的一次项
    :param c: 转移过程参数的常数项
    :param plot: 画图
    :param transition_mu: 4d, 外部输入量分布的均值,顺序：v_in, c_in, v_out, c_out
    :param transition_sigma: 4d，外部输入量的标准差
    :param L: 生成序列长度
    :return:
    """

    if transition_mu is None:
        transition_mu = [300, 0.3, 120, 0.68]
    if transition_sigma is None:
        transition_sigma = [30, 0.03, 10, 0.08]

    transition_mu = np.array(transition_mu)
    transition_sigma = np.diag(transition_sigma)


    title = ['round', 't', 'mass', 'pressure', 'v_in','c_in','v_out', 'c_out']

    all_data = []
    plt.figure(figsize=(12,8))
    for round in range(n):
        data = []

        mass = np.random.normal(begin_state_mu, begin_state_sigma)
        for t in range(L):
            """
            先生成input，再生成新的mass，最后得到观测的pressure
            """
            external_input = np.random.multivariate_normal(transition_mu, transition_sigma**2)
            v_in, c_in, v_out, c_out = tuple(external_input)

            # 将体积的单位转化为 10^2m^3，使得干砂质量单位为10^2t
            v_in *= 0.05
            v_out *= 0.05
            rho_in = a*c_in**2 + b**c_in + c
            rho_out = a*c_out**2 + b**c_out + c
            d_mass = v_in*c_in*rho_in - v_out*c_out*rho_out

            mass = mass + d_mass + np.random.normal(0, sigma)
            pressure = mass * D + pressure_bias + np.random.normal(0,delta)
            data.append(np.array([
                round, t, float(mass), float(pressure), v_in, c_in, v_out, c_out
            ]))

        data = np.array(data)
        all_data.append(data)


        if plot and round < 9:
            plt.subplot('33'+str(round+1))
            plt.title('round-'+str(round))
            plt.plot(data[:,2])
            plt.plot(data[:,3])
            plt.legend(['mass', 'pressure'])
            plt.ylabel(r'$20t$')
            plt.xlabel('min')
            plt.savefig(name + '.eps', dpi=500)


    if plot:
        plt.show()

    all_data = np.concatenate(all_data)
    #all_data[:,0:2] = np.array(all_data[:,0:2], dtype=int)


    df = pandas.DataFrame(all_data, columns=title)
    df['round'] = df['round'].astype(int)
    df['t'] = df['t'].astype(int)
    df.to_csv(name + '.csv')



if __name__ == '__main__':
    generate_fake_data(name='fake_train.', plot=True)
    generate_fake_data(name='fake_val', plot=True)
    generate_fake_data(name='fake_test', n=200, plot=True)






