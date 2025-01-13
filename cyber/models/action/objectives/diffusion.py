# diffusion.py implementation of the objective first proposed in https://arxiv.org/pdf/2303.04137
# formulation: end-to-end behavior cloning with diffusion models

import torch
import diffusers

from typing import Tuple, Optional, List


class DiffusionObjective:
    r"""DiffusionObjective class for handling **classic** diffusion objectives.

    This class contains boilerplate code for training and inference with a DDPM/DDIM model.

    DDPM scheduler handles "train on set training_steps = [0,1,...,n-1] steps,
    predict on prediction_steps \subset training_steps" while DDIM doesn't.

    DDIM scheduler handles "train on n steps, predict on m steps" better.
    """

    def __init__(self, scheduler_type: str, max_steps: int, **kwargs):
        """
        Args:
            scheduler_type(str): the type of scheduler to use. one of ['ddpm', 'ddim']
            max_steps(int): the maximum number of steps to take
            kwargs: additional arguments for the scheduler
        """
        if scheduler_type == "ddpm":
            self.scheduler = diffusers.DDPMScheduler(num_train_timesteps=max_steps, **kwargs)
        elif scheduler_type == "ddim":
            self.scheduler = diffusers.DDIMScheduler(num_train_timesteps=max_steps, **kwargs)

        self.scheduler_type = scheduler_type
        self.max_steps = max_steps

    def get_noise_noisy_actions(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            actions(torch.Tensor): the actions to add noise to. shape (batch_size, ac_chunk, ac_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: noisy_actions, noise, timesteps of
            shapes (batch_size, ac_chunk, ac_dim), (batch_size, ac_chunk, ac_dim), (batch_size,)
        """

        # create random timesteps and noise
        timesteps = torch.randint(0, self.max_steps, (actions.shape[0],))
        noise = torch.randn_like(actions)

        # add noise to actions
        noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)

        return noisy_actions, noise, timesteps

    def set_inference_timesteps(self, desired_inference_steps: Optional[int] = None, custom_inference_steps: Optional[List[int]] = None) -> torch.Tensor:
        """
        Set the inference timesteps for the scheduler.

        Args:
            model(torch.nn.Module): the model to generate actions from
            desired_inference_steps(int): the desired number of inference steps
            custom_inference_steps(List[int]): the custom inference steps to use,
                must be a descending subset of [0,1,...,max_steps-1]

        Returns:
            torch.Tensor: the generated actions. shape (batch_size, ac_chunk, ac_dim)

        Raises:
            ValueError: if both desired_inference_steps and custom_inference_steps are provided
            ValueError: if custom_inference_steps is used when scheduler_type is 'ddim'
        """
        if desired_inference_steps is not None and custom_inference_steps is not None:
            raise ValueError("Cannot provide both desired_inference_steps and custom_inference_steps")

        if custom_inference_steps is not None and self.scheduler_type == "ddim":
            raise ValueError("Cannot use custom_inference_steps with DDIM scheduler")

        # if neither are provided, use training steps as inference steps
        if desired_inference_steps is None and custom_inference_steps is None:
            self.scheduler.set_timesteps(self.max_steps)

        if custom_inference_steps is not None:
            self.scheduler.set_timesteps(custom_inference_steps)
        else:
            self.scheduler.set_timesteps(desired_inference_steps)

    def generate_actions(self, model: torch.nn.Module, model_input: dict, batch_size: int, act_dims: tuple) -> torch.Tensor:
        """
        Generate actions from the model.

        Args:
            model(torch.nn.Module): the model to generate actions from
            model_input(dict): the input to the model
            batch_size(int): the batch size of the input
            act_dims(tuple): the dimensions of the actions to generate

        Returns:
            torch.Tensor: the generated actions. shape (batch_size, ac_chunk, ac_dim)
        """
        model.eval()
        device = next(model.parameters()).device
        noise_actions = torch.randn(batch_size, *act_dims).to(device)  # sampled from N(0,1)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)

        for timestep in self.scheduler.timesteps:
            with torch.no_grad():
                model_input["noise_actions"] = noise_actions
                model_input["time_step"] = timestep.unsqueeze(0).repeat(batch_size).to(device)
                noise_pred = model(**model_input)
                noise_actions = self.scheduler.step(model_output=noise_pred, timestep=timestep, sample=noise_actions).prev_sample

        return noise_actions
