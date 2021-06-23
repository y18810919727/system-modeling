#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
import time
import torch
import hydra
import random
import pandas as pd
from control.utils import dict_to_Tensor
from common import detect_download
from control.scale import Scale
from matplotlib import pyplot as plt
from common import normal_interval

class ThickenerPressureSimulation:

    def __init__(self,
                 model,
                 dataset_path,
                 random_seed=None,
                 device='cpu',
                 ):

        """
        :param model: 建模实验中保存的模型
        :param random_seed: 随机化种子，会决定浓密机的初始状态和
        """
        self.dataset_path = dataset_path
        self.model = model
        self.device = device
        self.pressure = 0.0
        self.c_out = 0.0
        self.v_in = 0.0
        self.c_in = 0.0
        self.v_out = 0.0
        if random_seed is None:
            random_seed = random.randint(0, 1000000)
            print('random_seed:' + str(random_seed))
        self.random_seed = random_seed    # 随机种子作用
        self.simulation_time = 0      # 模拟仿真浓密机运行时间
        self.border = 0  # 绘图边界
        # scale = Scale(mean=np.zeros(5), std=np.ones(5))
        # 此处应该为另一个仿真数据集的归一化参数    ，测试用一致
        df_scale = pd.read_csv(os.path.join(os.getcwd(), "control/df_scale_cstr.csv"), encoding='utf-8').values
        self.s_mean = df_scale[:, 1]
        self.s_std = df_scale[:, 2]
        self.scale = Scale(self.s_mean, self.s_std)
        self.memory_state = None
        self.simulation_state_list = []
        self.figs_path = os.path.join(os.getcwd(), 'control/figs/', time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time())))
        self.set_value = np.array([100, 74, 236.9, 74, 236.9], dtype=np.float32)  # 设定值
        if not os.path.exists(self.figs_path):
            os.makedirs(self.figs_path)
        self.used_columns = ['pressure', 'c_out', 'v_in', 'c_in',
                             'v_out']

    def step(self, v_out_new: float):

        """

        1. 利用model预测下一时刻的pressure, c_out
        2. 从dataset取出新的v_in, c_in
        3. 更新浓密机hidden_state
        4. 返回五元组

        Args:
            v_out_new:

        Returns:

        """

        # 1 预测下一时刻的pressure, c_out, c_in , v_in  v_out_new.shape = (length, batch_size, 1)
        print(self.simulation_time)
        v_out_new = (v_out_new - self.s_mean[-1]) / self.s_std[-1]  # v_out 归一化
        self.v_out = v_out_new
        v_out_new = torch.from_numpy(np.array(v_out_new, dtype=np.float32)).to(self.device)
        v_out_new = v_out_new.unsqueeze(0).unsqueeze(0).unsqueeze(0)      # Tensor (1,1,1)
        pred_observations_dist, pred_observations_sample, new_memory_state = self.model.forward_prediction(v_out_new,  max_prob=True, memory_state=self.memory_state)
        pred_observations_sample = pred_observations_sample.cpu().detach()   # 应改为三输入二输出
        pred_observations_sample = pred_observations_sample.squeeze(0).squeeze(0).numpy()  # numpy([pressure,c_out,v_in,c_in])
        pred_sample = np.append(pred_observations_sample, self.v_out)
        pred_scale = self.scale.unscale_array(pred_sample)    # 数据反归一化  包括输入的v_out
        self.pressure, self.c_out, _, _, self.v_out = np.split(pred_scale, 5)   # part = tensor.shape[-1] =5
        self.pressure = float(self.pressure)
        self.c_out = float(self.c_out)
        self.v_out = float(self.v_out)

        # pred_observations_low, pred_observations_high = normal_interval(pred_observations_dist, 2)
        # 3 更新浓密机hidden_state ->估计因变量后验
        # _ = forward_posterior(external_input_new, pred_observations_sample, memory_state)
        self.memory_state = new_memory_state

        self.simulation_time = self.simulation_time + 1
        self.simulation_state_list.append(list(self.get_current_state().values()))
        if self.simulation_time % 50 == 0:       # 10周期绘图
            # 图1----仿真obs 和obs 设定值
            # pressure_set = [self.set_value[0] for x in range(self.border, self.simulation_time)]
            for pos in range(len(self.set_value)):
                set_pressure = []
                for i in range(200):
                    if i <= 100:
                        set_pressure.append(437)
                    elif i <= 200:
                        set_pressure.append(441)
                    elif i <= 300:
                        set_pressure.append(445)
                    else:
                        set_pressure.append(441)
                #plt.plot(range(self.border, self.simulation_time), set_pressure[self.border:self.simulation_time], label='set_value')
                plt.plot(range(self.border, self.simulation_time), [self.simulation_state_list[x][pos] for x in range(self.border, self.simulation_time)], label='planning')   # 仿真浓密机数据  -----暂时只选择pressure
                plt.title(self.used_columns[pos])
                plt.xlabel("time(minute)")
                plt.ylabel("Mud_Pressure(Pa)")
                plt.legend()
                plt.savefig(
                    os.path.join(self.figs_path, 'simulation_'+self.used_columns[pos]+'_'+str(self.border)+'_.png')
                )
                plt.close()

            #plt.plot(range(self.border, self.simulation_time),[self.set_value[pos] for x in range(self.border, self.simulation_time)], label='set_value')
            test_df = pd.DataFrame(self.simulation_state_list)
            test_df.to_csv(os.path.join(self.figs_path, "Simulation_list.csv"))

            self.border = self.simulation_time

            # 图3----仿真obs 分布和obs实际值
            # plt.plot(range(self.border, self.simulation_time), )

        # 4 返回五元组  (length, batch_size, 5)
        return self.get_current_state()

    def get_current_state(self):

        return {
            'pressure': self.pressure,
            'c_out': self.c_out,
            'v_in': self.v_in,
            'c_in': self.c_in,
            'v_out': self.v_out     # 改了一下顺序， 使得底流流量为最后一个
        }
