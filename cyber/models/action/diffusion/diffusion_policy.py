# diffusion_policy.py implementation of DiffusionPolicy class
# DiffusionPolicy is an ActionClass that uses diffusion backbone as its module and DDPMObjective as its objective.

import torch
from torch import nn

from cyber.models import CyberModule
from cyber.models.action.diffusion.backbones import DiffusionBackbone
from cyber.models.action.diffusion.objectives import DDPMObjective


class DiffusionPolicy(CyberModule):
    r"""DiffusionPolicy class for handling diffusion models.

    This class contains boilerplate code for training and inference with a diffusion model.
    """

    def __init__(self, backbone: DiffusionBackbone, num_train_timesteps: int, **kwargs):
        """
        Args:
            backbone(DiffusionBackbone): the diffusion backbone to use
            num_train_timesteps(int): the number of training timesteps to use
        """
        super().__init__()
        self.model = backbone
        self.objective = DDPMObjective(scheduler_type="ddim", max_steps=num_train_timesteps, **kwargs)

    def forward_method(self, noise_actions, condition, time_step, **kwargs):
        """
        Args:
            noise_actions(torch.Tensor): the noisy actions. shape (batch_size, ac_chunk, ac_dim)
            condition(torch.Tensor): the condition to use. shape (batch_size, obs_horizon, *condition_dim)
            time_step(torch.Tensor): the time step to use. shape (batch_size,)

        Returns:
            torch.Tensor: the predicted noise. shape (batch_size, action)
        """
        return self.model.forward(noise_actions=noise_actions, time_step=time_step, condition=condition, **kwargs)

    def compute_training_loss(self, actions: torch.Tensor, condition: torch.Tensor, **kwargs):
        """
        Args:
            actions(torch.Tensor): the actions to add noise to. shape (batch_size, ac_chunk, ac_dim)
            condition(torch.Tensor): the condition to use. shape (batch_size, obs_horizon, *condition_dim)

        Returns:
            torch.Tensor: the loss
        """
        noisy_actions, noise, timesteps = self.objective.get_noise_noisy_actions(actions)
        pred_noise = self.model.forward(noise_actions=noisy_actions, time_step=timesteps, condition=condition, **kwargs)
        return nn.functional.mse_loss(pred_noise, noise)

    def get_actions(self, condition: torch.Tensor, act_dims: int, **kwargs):
        """
        Args:
            condition(torch.Tensor): the condition to use. shape (batch_size, obs_horizon, *condition_dim)
            act_dims(int): the number of action dimensions to generate

        Returns:
            torch.Tensor: the predicted actions. shape (batch_size, ac_chunk, ac_dim)
        """
        self.model.set_condition_cache(condition)
        return self.objective.generate_actions(self.model, {}, batch_size=condition.shape[0], act_dims=(act_dims,))
