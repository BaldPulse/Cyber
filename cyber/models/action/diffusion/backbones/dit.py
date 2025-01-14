# dit.py diffusion transformer for robotics policy
# implementation of DiT-Block from 'The Ingredients for Robotic Diffusion Transformers' https://arxiv.org/abs/2410.10088
# by Dasari et al.

# This project uses dit-policy https://github.com/sudeepdasari/dit-policy which is licensed under the MIT License
# The original liscense is included in the dit_LICENSE.md file in the same directory as this file

# Code has been modified by adding comments, restructuring for clarity and adaptability to the project

# ORIGINAL NOTICE:
# Heavy inspiration taken from DETR by Meta AI (Carion et. al.): https://github.com/facebookresearch/detr
# and DiT by Meta AI (Peebles and Xie): https://github.com/facebookresearch/DiT

import copy
import logging

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from cyber.models.action.diffusion.backbones.nn_utils import SinusoidalPosEnc, FourierEmb
from cyber.models.action.diffusion.backbones.diffusionbackbone import DiffusionBackbone

logger = logging.getLogger(__name__)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _with_pos_embed(tensor, pos=None):
    return tensor if pos is None else tensor + pos


class _SelfAttnEncoder(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos):
        q = k = _with_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, value=src, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class _TransformerEncoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(base_module) for _ in range(num_layers)])

        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, src, pos):
        """
        Args:
            src: the source tensor
            pos: the positional encoding tensor. Optional: if None, no positional encoding is used
        """
        x, outputs = src, []
        for layer in self.layers:
            x = layer(x, pos)
            outputs.append(x)
        return outputs


class _TransformerDecoder(_TransformerEncoder):
    def forward(self, src, t, all_conds):
        x = src
        for layer, cond in zip(self.layers, all_conds, strict=False):
            x = layer(x, t, cond)
        return x


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        # self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) # never used
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = torch.mean(cond, axis=0)
        cond = cond + t

        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = x * scale[None] + shift[None]
        x = self.linear(x)
        return x.transpose(0, 1)  # because self-attention expects (seq_len, batch_size, hidden_dim)

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None] + self.shift(c)[None]

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scale.weight)
        nn.init.xavier_uniform_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # create modulation layers
        self.attn_mod1 = _ShiftScaleMod(d_model)
        self.attn_mod2 = _ZeroScaleMod(d_model)
        self.mlp_mod1 = _ShiftScaleMod(d_model)
        self.mlp_mod2 = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = torch.mean(cond, axis=0)
        cond = cond + t

        x2 = self.attn_mod1(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.attn_mod2(self.dropout1(x2), cond) + x

        x2 = self.mlp_mod1(self.norm2(x), cond)
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x2))))
        x2 = self.mlp_mod2(self.dropout3(x2), cond)
        return x + x2

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_mod1, self.attn_mod2, self.mlp_mod1, self.mlp_mod2):
            s.reset_parameters()


class DiTNoiseNet(DiffusionBackbone):
    """
    DiTNoiseNet class as proposed in https://arxiv.org/pdf/2410.10088

    This class conprises layers of self-attention encoder-decoder blocks with residual connections.
    It is designed to improve inference speed of diffusion through an encode-once-decode-many architecture.
    Training is stablized by applying conditions using adaLN-zero modulation layers instead of cross-attention.

    The design of the network can be considered an adaLN-zero conditioned decoder-only transformer with transformer encoders
    for conditions. While usable for non-diffusion objectives, the network is optimized for diffusion.

    """

    def __init__(
        self,
        ac_dim,
        ac_chunk,
        time_dim=256,
        hidden_dim=512,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
        activation="gelu",
    ):
        """
        Args:
            ac_dim: the dimension of the action representation
            ac_chunk: the number of actions to predict at once
            time_dim: the dimension of the time representation
            hidden_dim: the dimension of the token embeddings. Default: 512
            num_blocks: the number of transformer blocks (encoder and decoder). Default: 6
            dropout: the dropout rate. Default: 0.1
            dim_feedforward: the dimension of the feedforward network model. Default: 2048
            nhead: the number of heads in the multiheadattention models. Default: 8
            activation: the activation function of the model. Default: "gelu"
        """
        super().__init__()

        # positional encoding blocks
        self.enc_pos = SinusoidalPosEnc(hidden_dim)
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(ac_chunk, 1, hidden_dim), requires_grad=True),  # learnable decoder positional encoding
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # input encoder mlps
        self.time_net = self.out_net = nn.Sequential(FourierEmb(time_dim), nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),
        )

        # encoder blocks
        encoder_module = _SelfAttnEncoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.encoder = _TransformerEncoder(encoder_module, num_blocks)

        # decoder blocks
        decoder_module = _DiTDecoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        # turns predicted tokens into epsilons
        self.eps_out = _FinalLayer(hidden_dim, ac_dim)

        logger.info("number of diffusion parameters: {:e}".format(sum(p.numel() for p in self.parameters())))

    def set_condition_cache(self, condition: torch.Tensor):
        self.enc_cache = self.forward_enc(condition)

    def forward(self, noise_actions: torch.Tensor, time_step: torch.Tensor, condition: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        performs a forward pass through the DiTNoiseNet.
        If enc_cache is None, the encoder is run first.
        If enc_cache is not None, the encoder is skipped.

        Args:
            noise_actions (torch.Tensor): the noise actions. shape (batch_size, ac_chunk, ac_dim)
            time_step (torch.Tensor): the time steps. shape (batch_size,)
            obs_enc (torch.Tensor): the encoded observations. shape (batch_size, num_tokens, hidden_dim)

        Returns:
            enc_cache (list of tensors): the encoded cache. shape [num_blocks, (num_tokens, batch_size, hidden_dim)]
            the predicted epsilon actions. shape (ac_chunk, batch_size, ac_dim)

        if condition is None, then it is assumed that condition is cached somewhere in the model.
        if enc_cache is provided, it will overwride the cached enc_cache.
        """
        enc_cache = kwargs.get("enc_cache", None)
        if enc_cache is not None:
            self.enc_cache = enc_cache
        if condition is not None:
            self.enc_cache = self.forward_enc(condition)
        return self.forward_dec(noise_actions, time_step, self.enc_cache)

    def forward_enc(self, obs_enc: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            obs_enc (torch.Tensor): the encoded observations.  shape (batch_size, num_tokens, hidden_dim)

        Returns:
            the encoded cache (list of tensors). shape [num_blocks, (num_tokens, batch_size, hidden_dim)]
        """
        obs_enc = obs_enc.transpose(0, 1)  # because self-attention expects (seq_len, batch_size, hidden_dim)
        pos = self.enc_pos(obs_enc)
        enc_cache = self.encoder(obs_enc, pos)
        return enc_cache

    def forward_dec(self, noise_actions: torch.Tensor, time_step: torch.Tensor, enc_cache: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            noise_actions (torch.Tensor): the noise actions. shape (ac_chunk, batch_size, ac_dim)
            time_step (torch.Tensor): the time steps. shape (batch_size,)
            enc_cache (list of tensors): the encoded cache. shape [num_blocks, (num_tokens, batch_size, hidden_dim)]

        Returns:
            the predicted epsilon actions. shape (ac_chunk, batch_size, ac_dim)
        """
        time_enc = self.time_net(time_step)

        ac_tokens = self.ac_proj(noise_actions)
        ac_tokens = ac_tokens.transpose(0, 1)
        dec_in = ac_tokens + self.dec_pos

        # apply decoder
        dec_out = self.decoder(dec_in, time_enc, enc_cache)

        # apply final epsilon prediction layer
        return self.eps_out(dec_out, time_enc, enc_cache[-1])
