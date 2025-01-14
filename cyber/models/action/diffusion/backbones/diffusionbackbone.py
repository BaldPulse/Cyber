# diffusionbackbone.py base class for all diffusion backbones

import torch
from torch import nn
from abc import ABC, abstractmethod

from typing import Optional


class DiffusionBackbone(nn.Module, ABC):
    """DiffusionBackbone class for diffusion-based action models.

    This class is the base class for all diffusion backbones.
    It defines the forward pass signature for all diffusion backbones.
    """

    @abstractmethod
    def forward(
        self,
        noise_actions: torch.Tensor,
        time_step: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass of the diffusion backbone.

        Args:
            noise_actions (torch.Tensor): the noisy actions. shape (batch_size, *ac_shape)
            time_step (torch.Tensor): the time step. shape (batch_size,)
            condition (torch.Tensor): the condition. shape (batch_size, *condition_dim) (default: None)

        Returns:
            torch.Tensor: the predicted noise. shape (batch_size, *ac_shape)

        if condition is None, then it is assumed that condition is cached somewhere in the model.
        """
        raise NotImplementedError
