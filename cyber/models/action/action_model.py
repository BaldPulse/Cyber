# action_model.py
# Description: This file contains the ActionModel class, which is responsible for handling the action model.

import numpy as np

from torch import nn

from typing import Optional, List, ClassVar

from cyber.models import CyberModule


class ActionModel(CyberModule):
    """ActionModel class for learning-based policies.
    This class is the base class for all action models.

    The design of this class is inspired by HPT (Lirui Wang et al., 2024) and pi0 (Kevin Black et al., 2024).

    In essence, this class handles heterogeneities from two sources in two different ways:
    - embodiment heterogeneities: the heterogeneities in hardware is handled by switching between 'stems' and 'heads' (as in HPT)
    - task heterogeneities: the heterogeneities in tasks is handled by different prompts (as in pi0)

    If there are no heterogeneities, stems and trunks can be identity functions that does not accept any prompts and only
    the heads are used.

    TODO: make this class more flexible
    """

    module_registry: nn.ModuleDict = nn.ModuleDict()  # shared registry for all modules. this makes sharing modules easier
    module_info: ClassVar[dict[str, tuple[int, int, List[str]]]] = {}  # module_name: (num_params, num_trainable_params, used_in)

    def register_module(
        self,
        module: nn.Module,
        freeze_weights: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Register the module to the action model's module registry.

        Args:
            module (nn.Module): the module to register
            freeze_weights (bool): whether to freeze the weights of the module
            name (str): the name of the module to register. If None, the class name is used

        Raises:
            ValueError: if trying to register a module with the same name as an existing module
        """
        if name is None:
            name = module.__class__.__name__
        # check if the module is already registered
        if name in self.module_registry:
            raise ValueError(f"Module with name {name} is already registered. Delete it first.")
        # if freeze_weights, freeze the weights
        if freeze_weights:
            module.requires_grad_(False)
        # NOTE that if some weights are already frozen, this will not unfreeze them for you
        self.module_registry[name] = module

        # calculate the number of parameters as well as the number of trainable parameters
        num_params = sum(p.numel() for p in module.parameters())
        num_trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        self.module_info[name] = (num_params, num_trainable_params, [])

    def delete_module(self, name: str) -> None:
        """
        Delete the module from the action model's module registry.

        Args:
            name (str): the name of the module to delete
        """
        if name not in self.module_registry:
            raise ValueError(f"Module with name {name} is not registered.")
        del self.module_registry[name]
        del self.module_info[name]


class CyberAgent:
    def __init__(self, model: ActionModel):
        self.model = model

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return self.model.get_action(obs)

    # def train(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray) -> None:
    #     self.model.train(obs, action, reward, next_obs)
