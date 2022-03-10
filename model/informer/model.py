import torch
import torch.nn as nn
import torch.nn.functional as F

# from model.informer.utils.masking import TriangularCausalMask, ProbMask
from model.informer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from model.informer.decoder import Decoder, DecoderLayer
from model.informer.attn import FullAttention, ProbAttention, AttentionLayer
from model.informer.embed import DataEmbedding
from model.common import DiagMultivariateNormal as MultivariateNormal
from common import logsigma2cov


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, history_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                 output_attention=False, distil=True, mix=True):
        """
        Following the implementation in https://github.com/zhouhaoyi/Informer2020/blob/main/models/model.py

        Args:
            enc_in: input_size + output_size
            dec_in: input_size + output_size
            c_out: output_size
            history_len: The constant length of encoder's input.
            label_len: The length of decoder's input = label_len + out_len.
            out_len:  train_pred_len
            factor: probsparse attn factor
            d_model: dimension of model
            n_heads: num of heads
            e_layers: num of encoder layers
            d_layers:
            d_ff:
            dropout:
            attn: prob of full
            embed:
            activation:
            output_attention:
            distil:
            mix:
        """
        super(Informer, self).__init__()
        self.history_len = history_len
        self.label_len = label_len
        self.train_pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.input_size = enc_in
        self.observation_size = c_out
        self.dec_in = dec_in

        # Encoding
        self.enc_embedding = DataEmbedding(self.input_size + self.observation_size, d_model, embed, dropout)
        self.dec_embedding = DataEmbedding(self.dec_in, d_model, embed, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for _ in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):

        l, batch_size, _ = external_input_seq.size()

        history_seq = torch.zeros(
            (self.history_len, batch_size, self.input_size + self.observation_size), device=external_input_seq.device) \
            if memory_state is None else memory_state['history_seq']

        input_ob_seq = torch.cat([external_input_seq, observations_seq], dim=-1)
        history_seq = torch.cat([history_seq, input_ob_seq], dim=0)[-self.history_len:]
        # history_seq = history_seq

        enc_out = self.enc_embedding(history_seq.transpose(1, 0))
        enc_out, attns = self.encoder(enc_out)

        memory_state = {
            'enc_out': enc_out,
            'history_seq': history_seq
        }
        outputs = {
            'state_mu': observations_seq,
            # logsigma = -INF is equal to zero covariance matrix
            'state_logsigma': -torch.ones_like(observations_seq) * float('inf'),
        }
        return outputs, memory_state

    def forward_prediction(self, external_input_seq, max_prob=False, memory_state=None):

        l, batch_size, _ = external_input_seq.size()

        if memory_state is None:
            memory_state = self.forward_posterior(
                torch.zeros((self.history_len, batch_size, self.input_size), device=external_input_seq.device),
                torch.zeros((self.history_len, batch_size, self.observation_size), device=external_input_seq.device)
            )[-1]

        enc_out = memory_state['enc_out']
        history_seq = memory_state['history_seq']

        decoder_in = torch.cat([external_input_seq, torch.zeros((l, batch_size, self.observation_size),
                                                                device=external_input_seq.device).float()], dim=-1)
        decoder_in = torch.cat([history_seq, decoder_in], dim=0)[-(self.label_len + l):]

        decoder_out = self.dec_embedding(decoder_in.transpose(1, 0))
        decoder_out = self.decoder(decoder_out, enc_out)
        decoder_out = self.projection(decoder_out)
        predicted_seq = decoder_out.transpose(1, 0)[-l:]
        predicted_dist = MultivariateNormal(
            predicted_seq,
            torch.diag_embed(torch.ones_like(predicted_seq) * torch.Tensor([1e-6]).to(predicted_seq.device))
        )
        _, memory_state = self.forward_posterior(
            external_input_seq,
            predicted_seq,
            memory_state
        )
        outputs = {
            'predicted_dist': predicted_dist,
            'predicted_seq': predicted_seq
        }

        return outputs, memory_state

    def call_loss(self, external_input_seq, observations_seq, memory_state=None):

        assert external_input_seq.size(0) == observations_seq.size(0), \
            'The length of inputs_seq and observations_seq should be equal'
        assert external_input_seq.size(0) >= self.train_pred_len, \
            'The inputs_seq should be longer than model.train_pred_len'

        historical_input = external_input_seq[: -self.train_pred_len]
        historical_ob = observations_seq[:-self.train_pred_len]
        future_input = external_input_seq[-self.train_pred_len:]
        future_ob = observations_seq[-self.train_pred_len:]

        _, memory_state = self.forward_posterior(historical_input, historical_ob, memory_state)
        outputs, _ = self.forward_prediction(future_input, memory_state=memory_state)

        losses = {
            'loss': F.mse_loss(outputs['predicted_seq'], future_ob),
            'kl_loss': 0.0,
            'likelihood_loss': 0.0
        }
        return losses

    def decode_observation(self, outputs, mode='sample'):
        """

        Args:

            memory_state: forward_posterior之后返回的memory_state
            mode: dist or sample

        Returns:
            model为sample时，从分布采样(len,batch_size,observation)
            为dist时，直接返回分布对象torch.distributions.MultivariateNormal

        方法调用时不会给额外的输入参数，需在每次forward_prediction和forward_posterior之后将解码所需的信息存储在self里
        """

        if mode == 'dist':
            # state_logsigma is -INF logsigma2cov(state_logsigma) is a zero matrix.
            observations_normal_dist = MultivariateNormal(
                outputs['state_mu'],
                torch.diag_embed(torch.ones_like(outputs['state_mu']) * torch.Tensor(
                    [1e-6]).to(outputs['state_mu'].device))
            )
            return observations_normal_dist
        elif mode == 'sample':
            return outputs['state_mu']
