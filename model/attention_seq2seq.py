#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn

import torch.nn.functional as F

from model.common import DiagMultivariateNormal as MultivariateNormal

from common import softplus, inverse_softplus, cov2logsigma, logsigma2cov, split_first_dim, merge_first_two_dims
from model.func import normal_differential_sample, multivariate_normal_kl_loss

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        """
        References: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        Args:
            input_size:
            hidden_size:
            num_layers:
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_p=0.1, max_length=80):
        """
        References: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        Args:
            input_size:
            hidden_size:
            output_size:
            num_layers:
            dropout_p:
            max_length:
        """
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Linear(self.output_size + input_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[-1]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.transpose(1, 0))

        output = torch.cat((embedded[0], attn_applied[:, 0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = torch.tanh(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        return output, hidden, attn_weights

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class AttentionSeq2Seq(nn.Module):

    def __init__(self, input_size, observations_size, state_size=32, num_layers=1, dropout_p=0.1, max_length=80, label_length=40, train_pred_len=60):

        super(AttentionSeq2Seq, self).__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = state_size
        self.observations_size = observations_size
        self.drop_out = dropout_p
        self.max_length = max_length
        self.label_length = label_length
        self.train_pred_len = train_pred_len

        self.encoder = EncoderRNN(input_size + observations_size, state_size, num_layers)
        self.decoder = AttnDecoderRNN(input_size, state_size, observations_size, num_layers, dropout_p=dropout_p, max_length=max_length)

    def seq_encoding(self, seq_inputs, encoder_hidden, encoder_outputs_last):
        """

        Args:
            seq_inputs: concatenation of inputs and observations, with shape (..., batch_size, input_size + observation_size)
            encoder_hidden: initial hidden state with shape (1, batch_size, hidden_size)
            encoder_outputs_last: initial encoder_outputs (max_len, batch_size, hidden_size)

        Returns:
            new encoder_hidden: Feeding seq_inputs in RNN net and updating encoder_hidden
            new encoder_outputs: Concatenating the tail of encoder_outputs_last and new encoder_outputs according to seq_inputs

        """

        encoder_outputs_short, encoder_hidden = self.encoder(
            seq_inputs, encoder_hidden
        )
        # Joining the encoder_outputs_short at the tail of encoder_outputs_last
        encoder_outputs = torch.cat([encoder_outputs_last[-(self.max_length-encoder_outputs_short.size()[0]):], encoder_outputs_short], dim=0)
        return encoder_outputs, encoder_hidden

    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
        """

        Args:
            external_input_seq: 输入序列 (len, batch_size, input_size)
            observations_seq: 观测序列 (len, batch_size, ob_size)
            memory_state: 字典，模型记忆

        Returns: 隐变量均值,隐变量方差(确定性隐变量可以将其置0), 新的memory_state

        memory_state: {

        }
        把其他需要计算loss的变量存在对象里

        """
        l, batch_size, _ = external_input_seq.size()

        # Get the outputs of encoder, encoder_outputs -> (max_len, batch_size, hidden_size), encoder_hidden -> (1, batch_size, hidden_size)
        encoder_outputs, encoder_hidden, label_seq = (
            torch.zeros(self.max_length, batch_size, self.encoder.hidden_size, device=external_input_seq.device),
            self.encoder.init_hidden(batch_size, external_input_seq.device),
            torch.zeros(self.label_length, batch_size, self.observations_size+self.input_size, device=external_input_seq.device),
        ) if memory_state is None else (
            memory_state['encoder_outputs'], memory_state['encoder_hidden'], memory_state['label_seq']
        )

        # Updating the encoder_hidden_half and encoder_outputs_half according to the back half of input_ob_seq.

        input_ob_seq = torch.cat([external_input_seq, observations_seq], dim=-1)
        encoder_outputs, encoder_hidden = self.seq_encoding(input_ob_seq, encoder_hidden, encoder_outputs)
        memory_state = {}
        memory_state['encoder_outputs'] = encoder_outputs
        memory_state['encoder_hidden'] = encoder_hidden
        memory_state['label_seq'] = torch.cat(
            (label_seq, input_ob_seq), dim=0
        )[-self.label_length:]

        # Generating the outputs
        outputs = {
            'state_mu': observations_seq,
            # logsigma = -INF is equal to zero covariance matrix
            'state_logsigma': -torch.ones_like(observations_seq) * float('inf'),
        }

        return outputs, memory_state

    def forward_prediction(self, external_input_seq, max_prob=False, memory_state=None):
        """

        Args:
            external_input_seq: 输入序列 (len, batch_size, input_size)
            max_prob: 如果为False，从预测分布中随机采样，如果为True ,返回概率密度最大的估计值
            memory_state: 字典，模型记忆

        Returns: 元祖 (
            预测的观测序列分布 torch.distributions.MultivariateNormal,
            预测序列(序列怎么得到与max_prob有关) shape-> (len,batch_size,ob_size),
            模型记忆_ 字典
            )

        """

        l, batch_size, _ = external_input_seq.size()

        # Get the outputs of encoder, encoder_outputs -> (max_len, batch_size, hidden_size), encoder_hidden -> (1, batch_size, hidden_size)
        encoder_outputs, encoder_hidden, label_seq = (
            torch.zeros(self.max_length, batch_size, self.encoder.hidden_size, device=external_input_seq.device),
            self.encoder.init_hidden(batch_size, external_input_seq.device),
            torch.zeros(self.label_length, batch_size, self.observations_size+self.input_size, device=external_input_seq.device)

        ) if memory_state is None else (
            memory_state['encoder_outputs'], memory_state['encoder_hidden'], memory_state['label_seq'])

        decoder_hidden = encoder_hidden
        predicted_list = []

        # Feeding label_seq to generate decoder_hidden, following the implementation of process_one_batch in
        # https://github.com/zhouhaoyi/Informer2020/blob/main/exp/exp_informer.py/
        decoder_output = None
        for di in range(label_seq.size(0)):
            decoder_input = label_seq[di].unsqueeze(dim=0)
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs
            )

        # decoder_output = torch.zeros((batch_size, self.observations_size)).to(external_input_seq.device)
        for di in range(l):
            decoder_input = torch.cat([external_input_seq[di], decoder_output], dim=-1).unsqueeze(0)
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs
            )
            predicted_list.append(decoder_output)

        predicted_seq = torch.stack(predicted_list, dim=0)

        # The predicted distribution with zero covariance matrix (entropy is equal to 0)
        predicted_dist = MultivariateNormal(predicted_seq, torch.diag_embed(
            torch.ones_like(predicted_seq)/torch.Tensor([float(1e9)]).to(predicted_seq.device)
        ))

        input_ob_seq = torch.cat([external_input_seq, predicted_seq], dim=-1)

        # Updating encoder_outputs and encoder_hidden according to predicted outputs.
        encoder_outputs, encoder_hidden = self.seq_encoding(input_ob_seq, encoder_hidden, encoder_outputs)

        memory_state = {
            'encoder_outputs': encoder_outputs,
            'encoder_hidden':  encoder_hidden,
            'label_seq': torch.cat([label_seq, input_ob_seq], dim=0)[-self.label_length:]
        }

        outputs = {
            'predicted_dist': predicted_dist,
            'predicted_seq': predicted_seq
        }

        return outputs, memory_state

    def call_loss(self, external_input_seq, observations_seq, memory_state=None):
        """
        Args:
            external_input_seq:
            observations_seq:
            memory_state:
        Returns:
            三个标量: loss, kl_loss, decoding_loss， 没有后面两部分用0站位
            loss: 负对数似然loss
        loss要在batch_size纬度上取平均

        此处直接算被预测部分与observations_seq后一半之间的MSE作为loss
        """
        # l, batch_size, _ = self.predicted_seq.shape
        # loss = F.mse_loss(self.predicted_seq, self.observations_seq[-l:], reduction='mean')

        assert external_input_seq.size(0) == observations_seq.size(0), \
            'The length of inputs_seq and observations_seq should be equal'
        assert external_input_seq.size(0) >= self.train_pred_len, \
            'The inputs_seq should be longer than model.train_pred_len'

        historical_input = external_input_seq[: -self.train_pred_len]
        historical_ob = observations_seq[:-self.train_pred_len]
        future_input = external_input_seq[-self.train_pred_len:]
        future_ob = observations_seq[-self.train_pred_len:]

        _, memory_state = self.forward_posterior(historical_input, historical_ob, memory_state)
        outputs, memory_state = self.forward_prediction(future_input, memory_state=memory_state)
        losses = {
            'loss': F.mse_loss(outputs['predicted_seq'], future_ob),
            'kl_loss': 0.0,
            'likelihood_loss':0.0
        }
        return losses

    def decode_observation(self, outputs, mode='sample'):
        """

        Args:
            mode: dist or sample

        Returns:
            model为sample时，从分布采样(len,batch_size,observation)
            为dist时，直接返回分布对象torch.distributions.MultivariateNormal

        方法调用时不会给额外的输入参数，需在每次forward_prediction和forward_posterior之后将解码所需的信息存储在self里
        """

        if mode == 'dist':
            # state_logsigma is -INF logsigma2cov(state_logsigma) is a zero matrix.
            observations_normal_dist = MultivariateNormal(
                outputs['state_mu'], logsigma2cov(outputs['state_logsigma'])
            )
            return observations_normal_dist
        elif mode == 'sample':
            return outputs['state_mu']
