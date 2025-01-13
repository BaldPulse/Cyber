# nn_utils.py utilities for backbone models.

import torch
import numpy as np
from torch import nn


class SinusoidalPosEnc(nn.Module):
    """Sinusoidal Positional Encoding Module

    This class creates a sinusoidal positional encoding for the input tensor.
    The implementation matches the one described in the paper "Attention is All You Need" section 3.5.
    [https://arxiv.org/pdf/1706.03762]
    """

    def __init__(self, d_model, max_len=5000):
        """
        creates a sinusoidal positional encoding for the input tensor
        Args:
            d_model: the dimension of the input tensor
            max_len: the maximum length of the input tensor
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        inv_periods = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * inv_periods)
        pe[:, 1::2] = torch.cos(position * inv_periods)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor of shape (seq_len, batch_size, d_model) with positional encodings added
        """
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()


class SinusoidalTimestepEmb(nn.Module):
    """Sinusoidal Timestep Embedding Module

    Timestep k is turned into a positional encoding using the implementation in tensor2tensor
    [https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/modules/sinusoidal_positional_embedding.py#L15].

    but differs from Attention is All You Need. The positional encoding is then passed through a feedforward network to project to output dimensions.

    Note: the difference between this and SinusoidalPosEnc is because Ho et al. 2020's implementation was taken from Fairseq's implementation, which
    in turn was taken from tensor2tensor's implementation, which differs from Vaswani et al. 2017's implementation.
    """

    def __init__(self, time_dim, learnable_w=False):
        """
        Args:
            time_dim: the dimension of the input tensor, must be even
            learnable_w: whether the frequencies (fourier components) should be learnable (default: False uses "Attention if all you need" default)
        """
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

    def forward(self, x):
        assert len(x.shape) == 1, "assumes 1d input timestep array"
        x = x[:, None] * self.w[None]
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)
        return x
