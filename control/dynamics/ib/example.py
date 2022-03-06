# coding=utf-8
from control.dynamics.ib.IDS import IDS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
The MIT License (MIT)

Copyright 2017 Siemens AG

Author: Stefan Depeweg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

n_trajectories = 6
T = 10000

data = np.zeros((n_trajectories,T))
data_cost = np.zeros((n_trajectories,T))

# 扩充训练集
# 自定义at: [0.5, 1, 0, -1, -0.5]
bt = [(2 * np.random.rand(3) -1)/2 for x in range(0,2000)]+\
     [((2 * np.random.rand(3) -1)/2 + 0.5) for x in range(0,2000)]+\
     [(2 * np.random.rand(3) -1)/2 for x in range(0,2000)]+\
     [((2 * np.random.rand(3) -1)/2 - 0.5) for x in range(0,2000)]+\
     [(2 * np.random.rand(3) -1)/2 for x in range(0,2000)]
# bt = [[0, 0, 0] for x in range(0,10000)]
bt = np.array(bt)

for k in range(n_trajectories):
    obs = []
    action = []
    env = IDS( p=100)
    for t in range(T):
        at = 2 * np.random.rand(3) -1     #  [-1,1]
        markovStates = env.step(at)
        action.append(np.array(at))
        obs.append(np.array([env.state[k] for k in env.observable_keys]))
        data[k,t] = env.visibleState()[-1]
    df_obs = pd.DataFrame(obs, columns=env.observable_keys)
    df_action = pd.DataFrame(action, columns=['delta_v','delta_g','delta_h'])
    df_state = pd.concat([df_action, df_obs],axis=1)
    df_state.to_csv("ibState%s.csv" % k, index=False)
plt.plot(data.T)
plt.xlabel('T')
plt.ylabel('Reward')
plt.show()