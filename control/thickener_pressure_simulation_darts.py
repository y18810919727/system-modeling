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
from darts import TimeSeries
from darts.models import RNNModel, BlockRNNModel

class ThickenerPressureSimulation:

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

        """
        :param model: 建模实验中保存的模型
        :param random_seed: 随机化种子，会决定浓密机的初始状态和
        """
        self.dataset_path = dataset_path
        self.model = model
        self.device = device
        self.observation = [0.0 for x in range(output_dim)]
        self.action = [0.0 for x in range(input_dim)]
        self.obs_name = used_columns[input_dim:]
        self.action_name = used_columns[:input_dim]
        self.set_value = set_value
        if random_seed is None:
            random_seed = random.randint(0, 1000000)
            print('random_seed:' + str(random_seed))
        self.random_seed = random_seed    # 随机种子作用
        self.simulation_time = 0      # 模拟仿真浓密机运行时间
        self.border = 0  # 绘图边界
        '''归一化
        # 此处应该为另一个仿真数据集的归一化参数    ，测试用一致
        df_scale = pd.read_csv(os.path.join(os.getcwd(), "control/df_scale.csv"), encoding='utf-8').values
        self.s_mean = df_scale[:, 1]
        self.s_std = df_scale[:, 2]
        self.scale = Scale(self.s_mean, self.s_std)
        '''
        self.memory_state = None
        self.simulation_state_list = []
        self.figs_path = figs_path

    def step(self, planning_action):

        """

        1. 利用model预测下一时刻的pressure, c_out
        2. 从dataset取出新的v_in, c_in
        3. 更新浓密机hidden_state
        4. 返回五元组

        Args:
            planning_action:

        Returns:

        """

        # 1 预测 output  input.shape = (length, batch_size, input_dim)
        print(self.simulation_time)
        '''
        v_out_new = (v_out_new - self.s_mean[-1]) / self.s_std[-1]  # v_out 归一化
        self.v_out = v_out_new
        v_out_new = torch.from_numpy(np.array(v_out_new, dtype=np.float32)).to(self.device)
        v_out_new = v_out_new.unsqueeze(0).unsqueeze(0).unsqueeze(0)      # Tensor (1,1,1)
        '''
        external_input_seq = torch.tensor(planning_action, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
        # output, _ = self.model.forward_prediction(, max_prob=True, memory_state=memory_state)
        output, new_memory_state = self.model.forward_prediction(external_input_seq,  max_prob=True, memory_state=self.memory_state)   # 还使用forward_prediction()吗
        pred_observations_sample = output['predicted_seq']
        pred_observations_sample = pred_observations_sample.cpu().detach()
        pred_observations_sample = pred_observations_sample.squeeze(0).squeeze(0).numpy()
        self.observation = pred_observations_sample.tolist()
        self.action = planning_action
        '''
        pred_sample = np.append(pred_observations_sample, self.v_out)
        # pred_scale = self.scale.unscale_array(pred_sample)    # 数据反归一化  包括输入的v_out
        # self.pressure, self.c_out, _, _, self.v_out = np.split(pred_scale, 5)   # part = tensor.shape[-1] =5
        self.pressure, self.c_out, _, _, self.v_out = np.split(pred_sample, 5)  # part = tensor.shape[-1] =5
        self.pressure = float(self.pressure)
        self.c_out = float(self.c_out)
        self.v_out = float(self.v_out)
        '''
        # pred_observations_low, pred_observations_high = normal_interval(pred_observations_dist, 2)
        # 3 更新浓密机hidden_state ->估计因变量后验
        # _ = forward_posterior(external_input_new, pred_observations_sample, memory_state)
        self.memory_state = new_memory_state

        self.simulation_time = self.simulation_time + 1
        self.simulation_state_list.append(list(self.get_current_state().values()))

        if self.simulation_time % 20 == 0:       # 10周期绘图
            # 图1----仿真obs 和obs 设定值
            for pos in range(0, len(self.observation)):
                # plt.plot(range(self.border, self.simulation_time), [self.set_value[pos] for x in range(self.border, self.simulation_time)], label='set_value')
                plt.plot(range(self.border, self.simulation_time), [self.simulation_state_list[x][0][pos] for x in range(self.border, self.simulation_time)], label='planning')   # 仿真浓密机数据  -----暂时只选择pressure
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
                plt.close()

            # plt.plot(range(self.border, self.simulation_time),[self.set_value[pos] for x in range(self.border, self.simulation_time)], label='set_value')
            # test_df = pd.DataFrame(self.simulation_state_list[0])
            # test_df.to_csv(os.path.join(self.figs_path, "Simulation_list.csv"),index=False)

            # 画动作序列的图
            for pos in range(0, len(self.action)):
                # plt.plot(range(self.border, self.simulation_time), [self.set_value[pos] for x in range(self.border, self.simulation_time)], label='set_value')
                plt.plot(range(self.border, self.simulation_time),
                         [self.simulation_state_list[x][1][pos] for x in range(self.border, self.simulation_time)],
                         label='planning' + str(pos))  # 仿真浓密机数据  -----暂时只选择pressure
                plt.title(self.action_name[pos])
                plt.xlabel("time(minute)")
                plt.ylabel("control_value")
                plt.legend()
                try:
                    plt.savefig(
                        os.path.join(self.figs_path, 'action_' + str(pos) + '_' + str(self.border) + '_.png')
                    )
                except Exception as e:
                    # import pdb
                    # pdb.set_trace()
                    raise e
                plt.close()

            self.border = self.simulation_time
            # plt.plot(range(self.border, self.simulation_time),[self.set_value[pos] for x in range(self.border, self.simulation_time)], label='set_value')
            # test_df = pd.DataFrame(self.simulation_state_list[0])
            # test_df.to_csv(os.path.join(self.figs_path, "Simulation_list.csv"),index=False)

        # 4 返回五元组  (length, batch_size, 5)
        return self.get_current_state()

    def get_current_state(self):

        return {
            'observation': self.observation,
            'action': self.action,
        }