# diffusion_policy.py implementation of DiffusionPolicy class
# DiffusionPolicy is an ActionClass that uses diffusion backbone as its module and DDPMObjective as its objective.

from cyber.models.action import ActionModel
from cyber.models.action.diffusion.backbones import DiffusionBackbone
from cyber.models.action.diffusion.objectives import DDPMObjective


class DiffusionPolicy(ActionModel):
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
        self.register_module("backbone", backbone)
        self.register_domain("main", "none", "none", "backbone")
        self.objective = DDPMObjective(scheduler_type="ddim", max_steps=num_train_timesteps, **kwargs)
