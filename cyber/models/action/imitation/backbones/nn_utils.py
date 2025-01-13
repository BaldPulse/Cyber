# nn_utils.py utilities for backbone models.

import torch
import numpy as np
from torch import nn


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal Embedding class

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
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
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
