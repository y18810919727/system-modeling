#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from control.pressure_control_service import control_service_start

from control.utils import my_JSON_serializable, dict_to_Tensor

import requests
import time
import argparse
from control.thickener_pressure_simulation import ThickenerPressureSimulation

parser = argparse.ArgumentParser('Pressure Control Test')
parser.add_argument('-R',  type=int, default=200, help="Rounds for Test")
parser.add_argument('--simulation', type=str, default='control/model/test.pkl', help='ckpt path of simulation model.')
parser.add_argument('--planning',  type=str, default='control/model/test.pkl', help="ckpt path of planning model.")
parser.add_argument('--ip',  type=str, default='localhost', help="ckpt path of planning model.")
parser.add_argument('-v', '--vis', action='store_true', default=False)
parser.add_argument('--service', action='store_true', default=False)
parser.add_argument('-cuda',  type=int, default=3, help="GPU ID")
parser.add_argument('--length',  type=int, default=50, help="The length of optimized sequence for planning")
parser.add_argument('--num_samples',  type=int, default=32, help="The number of samples in CEM planning")
parser.add_argument('--num_iters',  type=int, default=32, help="The number of iters in CEM planning")
parser.add_argument('-r', '--random_seed',  type=int, default=1, help="Random seed in experiment")
parser.add_argument('--port',  type=int, default=6006, help="The number of iters in CEM planning")
parser.add_argument('--debug', action='store_true', default=False)
config = parser.parse_args()


def get_ob_list(cur_ob_dict):
    cur_ob = [
        cur_ob_dict['pressure'],
        cur_ob_dict['v_in'],
        cur_ob_dict['v_out'],
        cur_ob_dict['c_in'],
        cur_ob_dict['pressure'],
        cur_ob_dict['v_in'],
        cur_ob_dict['v_out'],
        cur_ob_dict['c_in'],
        cur_ob_dict['c_out']
    ]
    return cur_ob


def main(args):
    device = torch.device("cuda:{}".format(str(args.cuda)) if torch.cuda.is_available() else "cpu")
    if args.service:
        """
        没调通，暂时不要用service模式。先手动开service，在开test。
        """

        from torch.multiprocessing import Process, set_start_method
        # from multiprocessing import Process
        # set_start_method('spawn')
        # set_start_method('spawn', force=True)
        control_service = Process(target=control_service_start, args=(args,))
        control_service.start()
        time.sleep(8)

    model = torch.load(args.simulation)
    model.to(device)
    simulated_thickener = ThickenerPressureSimulation(model=model,
                                                      dataset_path='./data/part', random_seed=args.random_seed)

    # region 先根据浓密机当前数据得到初始的memory_state
    resp = requests.post(
        "http://{}:{}/update".format(args.ip, str(args.port)),
        data={
            'monitoring_data': json.dumps(get_ob_list(simulated_thickener.get_current_state())),
            'memory_state': None
        })
    memory_state_json = resp.json()['memory_state']
    # endregion
    for _ in range(args.R):

        # region 请求执行cem-planning
        resp = requests.post("http://{}:{}/cem-planning".format(args.ip, str(args.port)), data={
            'memory_state': memory_state_json
        })
        print('CEM-planning - {}'.format(resp.json()))
        planning_v_out = float(resp.json()['planning_v_out'][0])  # response中的planning_v_out为list
        # endregion

        # 将控制结果应用于仿真浓密机
        simulated_thickener.step(planning_v_out)

        # region 获得浓密机最新状态，并更新memory-state
        resp = requests.post("http://{}:{}/update".format(args.ip, str(args.port)), data={
            'monitoring_data': json.dumps(get_ob_list(simulated_thickener.get_current_state())),
            'memory_state': memory_state_json,
        })
        memory_state_json = resp.json()['memory_state']
        print('update - resp={}'.format(resp.json()))
        # endregion

        # print('send: {}'.format(my_JSON_serializable(memory_state_json)))
        # print('resp: {}'.format(resp.json()))

        time.sleep(3) # 测试用，后期跑可以删掉


if __name__ == '__main__':
    main(config)
