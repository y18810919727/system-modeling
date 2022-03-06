#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import json
import time
import torch
import hydra
import random
import pandas
from control.utils import dict_to_Tensor
from common import detect_download
from control.scale import Scale
from matplotlib import pyplot as plt
from control.dynamics.ib.IDS import IDS
import numpy as np

class IndustrialBenchmarkSimulation:

    def __init__(self,
                 set_value,
                 input_dim,
                 output_dim,
                 used_columns,
                 figs_path,
                 model,
                 dataset_path,
                 random_seed=None,
                 device='cpu',
                 ):

        self.env = IDS(p=100)
        self.model = model
        self.dataset_path = dataset_path
        self.device = device
        self.observation = [0.0 for x in range(output_dim)]
        self.delta_action = [0.0 for x in range(input_dim)]
        self.action = [0.0 for x in range(input_dim)]
        self.set_value = set_value
        if random_seed is None:
            random_seed = random.randint(0, 1000000)
            print('random_seed:' + str(random_seed))
        self.random_seed = random_seed    # 随机种子作用
        self.simulation_time = 0      # 仿真运行时间
        self.border = 0  # 绘图边界
        self.obs_name = used_columns[input_dim:]
        self.action_name = used_columns[:input_dim]
        self.simulation_state_list = []
        # self.figs_path = os.path.join(os.getcwd(), 'control/figs/', time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time())))
        # if not os.path.exists(self.figs_path):
        #     os.makedirs(self.figs_path)
        self.figs_path = figs_path

    def step(self, planning_action, planning_delta_action):

        """

        1. 利用model预测下一时刻的pressure, c_out
        2. 从dataset取出新的v_in, c_in
        3. 更新浓密机hidden_state
        4. 返回五元组

        Args:
            planning_delta_action:

        Returns:

        """

        # 1 预测 output  input.shape = (length, batch_size, input_dim)
        print(self.simulation_time)

        # # at 反归一化 函数
        # at_mean = np.array(self.model.mean.values.tolist()[1:4])
        # at_std = np.array(self.model.std.values.tolist()[1:4])
        #
        # # 转array矩阵乘,计算过后转回list
        # planning_delta_action = np.array(planning_delta_action) * at_std + at_mean
        # planning_action = planning_action.tolist()

        obs = []
        # at = 2 * np.random.rand(3) - 1  # [-1,1]
        self.action = planning_action
        self.delta_action = planning_delta_action # ib env的at表示的是在基础值50的变动值
        self.env.step(self.delta_action)  # 画图的时候可以查看一下返回值，Gym库会有各种返回值。
        obs.append(np.array([self.env.state[k] for k in self.env.observable_keys])) # list

        # df_obs = pandas.DataFrame(obs, columns=env.observable_keys)
        # df_obs.to_csv("ibState%s.csv" % k, index=False)

        self.observation = [obs[0][4],obs[0][5],obs[0][7]]
        self.simulation_time = self.simulation_time + 1
        self.simulation_state_list.append(list(self.get_current_state().values()))

        if self.simulation_time % 20 == 0:       # 10周期绘图
            # 图1----仿真obs 和obs 设定值
            for pos in range(0, len(self.observation)):
                # plt.plot(range(self.border, self.simulation_time), [self.set_value[pos] for x in range(self.border, self.simulation_time)], label='set_value')
                plt.plot(range(self.border, self.simulation_time), [self.simulation_state_list[x][0][pos] for x in range(self.border, self.simulation_time)], label='planning'+str(pos))   # 仿真浓密机数据  -----暂时只选择pressure
                plt.title(self.obs_name[pos])
                plt.xlabel("time(minute)")
                plt.ylabel("control_value")
                plt.legend()
                try:
                    plt.savefig(
                        os.path.join(self.figs_path, 'simulation_'+str(pos)+'_'+str(self.border)+'_.png')
                    )
                except Exception as e:
                    # import pdb
                    # pdb.set_trace()
                    raise e
                # plt.close()
            # 画动作序列的图
            for pos in range(0, len(self.delta_action)):
                # plt.plot(range(self.border, self.simulation_time), [self.set_value[pos] for x in range(self.border, self.simulation_time)], label='set_value')
                plt.plot(range(self.border, self.simulation_time), [self.simulation_state_list[x][1][pos] for x in range(self.border, self.simulation_time)], label='planning'+str(pos))   # 仿真浓密机数据  -----暂时只选择pressure
                plt.title(self.action_name[pos])
                plt.xlabel("time(minute)")
                plt.ylabel("control_value")
                plt.legend()
                try:
                    plt.savefig(
                        os.path.join(self.figs_path, 'action_'+str(pos)+'_'+str(self.border)+'_.png')
                    )
                except Exception as e:
                    # import pdb
                    # pdb.set_trace()
                    raise e
                plt.close()


            # plt.plot(range(self.border, self.simulation_time),[self.set_value[pos] for x in range(self.border, self.simulation_time)], label='set_value')
            test_df = pandas.DataFrame(self.simulation_state_list)
            test_df.to_csv(os.path.join(self.figs_path, "Simulation_list.csv"),index=False)

            self.border = self.simulation_time

        # monitor_data 归一化
        obs_mean = np.array(
            [self.model.mean.values.tolist()[4], self.model.mean.values.tolist()[5], self.model.mean.values.tolist()[-1]])
        obs_std = np.array(
            [self.model.std.values.tolist()[4], self.model.std.values.tolist()[5], self.model.std.values.tolist()[-1]])

        self.observation = (np.array(self.observation) - obs_mean) / obs_std
        self.observation = self.observation.tolist()
        # self.delta_action = (np.array(self.delta_action) - at_mean) / at_std
        # self.delta_action = self.delta_action.tolist()

        # 4 返回五元组  (length, batch_size, 5)
        return self.get_current_state()

    def get_current_state(self):

        return {
            'observation': self.observation,
            'action': self.action,
        }