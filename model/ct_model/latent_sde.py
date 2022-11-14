#!/usr/bin/python
# -*- coding:utf8 -*-


import torch
from torch import nn
from model.common import DBlock
from common import logsigma2cov, split_first_dim, merge_first_two_dims, softplus
from model.common import DiagMultivariateNormal as MultivariateNormal, MLP, _stable_division
from model.ct_model.ct_common import dt2ts
from model.func import normal_differential_sample, multivariate_normal_kl_loss, kld_gauss
from einops import rearrange
from model.ct_model import ODEFunc
from model.ct_model.diffeq_solver import solve_diffeq
from model.ct_model.interpolation import CubicInterpolation, KernelInterpolation, ZeroInterpolation

import torch
from torch import distributions, nn, optim

import torchsde
sdeint_fn = torchsde.sdeint

from common import LinearScheduler, EMAMetric

class LatentSDE(torchsde.SDEIto):

    def __init__(self, h_size, y_size, u_size, theta=1.0, mu=0.0, sigma=0.5, inter='gp',
                 dt=1e-2, rtol=1e-3, atol=1e-3, method='euler', adaptive=False):
        super(LatentSDE, self).__init__(noise_type="diagonal")

        u_size -= 1 # The last dimension of input variable represents dt
        self.y_size, self.u_size, self.h_size, self.inter = y_size, u_size-1, h_size, inter
        self.dt, self.rtol, self.atol, self.method, self.adaptive = dt, rtol, atol, method, adaptive

        # Prior drift.
        adaptive_theta = torch.nn.Parameter(torch.tensor([[theta]]), requires_grad=True)
        self.register_buffer("theta", adaptive_theta)
        self.mu = torch.nn.Sequential(
            torch.nn.Linear(h_size + u_size, 2 * h_size),
            torch.nn.Tanh(),
            torch.nn.Linear(2 * h_size, h_size)
        )

        # p(y0).
        self.register_buffer("py0_mean", nn.Parameter(torch.zeros((1, h_size)), requires_grad=True))
        self.register_buffer("py0_logvar", nn.Parameter(torch.zeros((1, h_size)), requires_grad=True))

        # Approximate posterior drift: Takes in 2 positional encodings and the state.
        self.net = nn.Sequential(
            nn.Linear(h_size+y_size+u_size, 2*h_size),
            nn.Tanh(),
            nn.Linear(2*h_size, h_size)
        )
        # q(y0).
        self.qy0_mean = nn.Parameter(torch.zeros((1, h_size)), requires_grad=True)
        self.qy0_logvar = nn.Parameter(torch.zeros((1, h_size)), requires_grad=True)

        # Common diffusion.
        self.sigma = torch.nn.Sequential(
            torch.nn.Linear(h_size + u_size, 2 * h_size),
            torch.nn.Tanh(),
            torch.nn.Linear(2 * h_size, h_size),
            nn.Softplus()
        )

        # Initialization trick from Glow.
        self.net[-1].weight.data.fill_(0.)
        self.net[-1].bias.data.fill_(0.)

        self.decoder = DBlock(h_size, 2*h_size, y_size)

        self.u_inter, self.y_inter = None, None

    def interpolate(self, ts, x):
        if self.inter == 'gp':
            Interpolation = KernelInterpolation
        elif self.inter == 'cubic':
            Interpolation = CubicInterpolation
        elif self.inter =='zero':
            Interpolation = ZeroInterpolation

        return Interpolation(ts, x)

    def update_u(self, ts, u):
        self.u_inter = self.interpolate(ts, u)

    def update_y(self, ts, y):
        self.y_inter = self.interpolate(ts, y)

    def f(self, t, h):  # Approximate posterior drift.

        return self.net(torch.cat([h, self.u_inter(t), self.y_inter(t)], dim=-1))

    def g(self, t, h):  # Shared diffusion.
        return self.sigma(torch.cat([h, self.u_inter(t)], dim=-1))

    def h(self, t, h):  # Prior drift
        target = self.mu(torch.cat([h, self.u_inter(t)], dim=-1))
        return self.theta * (target - h)

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, :self.h_size]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        h = y[:, :self.h_size]
        g = self.g(t, h)
        g_logqp = torch.zeros_like(y[:, self.h_size:])
        return torch.cat([g, g_logqp], dim=1)

    def forward(self, ts, us, ys, batch_size, eps=None):
        eps = torch.randn(batch_size, self.h_size).to(self.qy0_std) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)  # KL(t=0).

        self.update_u(ts, us)
        self.update_y(ts, ys)
        aug_y0 = torch.cat([y0, torch.zeros(batch_size, 1).to(y0)], dim=1)
        aug_ys = sdeint_fn(
            sde=self,
            y0=aug_y0,
            ts=ts[:, 0],
            method=self.method,
            dt=self.dt,
            adaptive=self.adaptive,
            rtol=self.rtol,
            atol=self.atol,
            names={'drift': 'f_aug', 'diffusion': 'g_aug'}
        )
        # aug_ys = sdeint_fn_batched(self, aug_y0, ts, sdeint_fn, method=args.method, dt=args.dt, adaptive=args.adaptive,
        #                   rtol=args.rtol, atol=args.atol, names={'drift': 'f_aug', 'diffusion': 'g_aug'})
        ys, logqp_path = aug_ys[:, :, :self.h_size], aug_ys[-1, :, -1]
        logqp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
        return ys, logqp

    def sample_p(self, ts, us, batch_size, y0=None, eps=None, bm=None):
        self.update_u(ts, us)
        eps = torch.randn(batch_size, self.h_size).to(self.py0_mean) if eps is None else eps
        y0 = self.py0_mean + eps * self.py0_std if y0 is None else y0
        return sdeint_fn(self, y0, ts[:, 0], bm=bm, method=self.method, dt=self.dt, names={'drift': 'h'})
        # return sdeint_fn_batched(self, y0, ts, sdeint_fn, method=args.method, dt=args.dt, names={'drift': 'h'})

    def sample_q(self, ts, us, ys, batch_size, y0=None, eps=None, bm=None):
        self.update_u(ts, us)
        self.update_y(ts, ys)
        eps = torch.randn(batch_size, self.h_size).to(self.qy0_mean) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std if y0 is None else y0
        return sdeint_fn(self, y0, ts[:, 0], bm=bm, method=self.method, dt=self.dt)
        # return sdeint_fn_batched(self, y0, ts, sdeint_fn, method=args.method, dt=args.dt)

    @property
    def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)


    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
        """
        训练时：估计隐变量后验分布，并采样，用于后续计算模型loss
        测试时: 为后续执行forward_prediction计算memory_state(h, rnn_hidden)
        Args:
            external_input_seq: 系统输入序列(进出料浓度、流量) (len, batch_size, input_size)
            observations_seq: 观测序列(泥层压强) (len, batch_size, observations_size)
            memory_state: 模型支持长序列预测，之前forward过程产生的记忆信息压缩在memory_state中

        Returns:

        """

        external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]
        l, batch_size, _ = external_input_seq.size()
        y0 = memory_state['y'] if memory_state is not None else None
        ts = dt2ts(dt)
        ys = self.sample_q(ts, external_input_seq, observations_seq, batch_size, y0=y0, eps=None, bm=None)
        outputs = {
            'state_mu': ys,
            'state_logsigma': torch.zeros_like(ys)-1e8,
            'sampled_state': ys,
        }
        return outputs, {'y': ys[-1]}

    def forward_prediction(self, external_input_seq, n_traj=16, max_prob=False, memory_state=None):

        with torch.no_grad():

            l, batch_size, _ = external_input_seq.size()
            external_input_seq = external_input_seq.repeat(1, n_traj, 1)
            external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]
            ts = dt2ts(dt)
            device = external_input_seq.device
            y0 = memory_state['y'] if memory_state is not None else None
            y0 = y0.repeat(n_traj, 1)
            ys = self.sample_p(ts, external_input_seq, batch_size=batch_size, y0=y0, eps=0, bm=None)
            mean, logsigma = self.decoder(ys)

            predicted_seq_sample = normal_differential_sample(
                MultivariateNormal(mean, logsigma2cov(logsigma))
            )
            predicted_seq_sample = rearrange(predicted_seq_sample, 'l (n b) d -> l n b d', n=n_traj, b=batch_size).permute(0, 2, 1, 3).contiguous()
            predicted_seq = torch.mean(predicted_seq_sample, dim=2)
            predicted_dist = MultivariateNormal(
                predicted_seq, torch.diag_embed(predicted_seq_sample.var(dim=2))
                # 此处如何生成分布(如何提取均值和方差)
            )
            outputs = {
                'predicted_dist': predicted_dist,
                'predicted_seq': predicted_seq,
                'predicted_seq_sample': predicted_seq_sample
            }
            y = ys[-1, :batch_size]
        return outputs, {'y': y}

    def call_loss(self, external_input_seq, observations_seq, memory_state=None):
        l, batch_size, _ = observations_seq.shape
        external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]
        ts = dt2ts(dt)
        # def forward(self, ts, us, ys, batch_size, eps=None):

        ys, kl = self.forward(ts, external_input_seq, observations_seq, batch_size, eps=None)
        mean, logstd = self.decoder(ys)
        likelihood = MultivariateNormal(mean, logsigma2cov(logstd))
        logpy = likelihood.log_prob(observations_seq).sum(dim=0).mean(dim=0)
        return {
            'loss': kl - logpy,
            'kl_loss': kl,
            'likelihood_loss': -logpy
        }

    def decode_observation(self, outputs, mode='sample'):

        """

        Args:
            state: with shape (len, batch_size, state_size)
            mode: dist or sample

        Returns:

        """
        mean, logsigma = self.decoder(outputs['sampled_state'])
        observations_normal_dist = MultivariateNormal(
            mean, logsigma2cov(logsigma)
        )
        if mode == 'dist':
            return observations_normal_dist
        elif mode == 'mean':
            return mean
        elif mode == 'sample':
            return observations_normal_dist.sample()
