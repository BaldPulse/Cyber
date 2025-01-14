# action_model.py
# Description: This file contains the ActionModel class, which is responsible for handling the action model.

# The ActionModel class is heavily inspired by the HPT project https://github.com/liruiw/HPT without
# using the HPT codebase.

import numpy as np

import torch
from torch import nn


from typing import Optional, List, ClassVar, Dict

import prettytable
import logging

from cyber.models import CyberModule

# class ModuleQueue:
#     '''ModuleQueue class for handling queues for modules.
#     '''

#     def __init__(self, batchsize: int):
#         self.batchsize = batchsize
#         self.Q = thmp.Queue()
#         self.buffer = None

#     def enqueue(self, data: Dict[str, torch.Tensor]) -> None:
#         '''Enqueue the data to the queue.

#         Args:
#             data (Dict[str, torch.Tensor]): the data to enqueue
#         '''
#         # 1. get the length of the data
#         length = len(data[list(data.keys())[0]])
#         # 2. put the data into the queue
#         # 2.1 the buffer


class ActionModel(CyberModule):
    """ActionModel class for learning-based policies.
    This class is the base class for all action models.

    The design of this class is inspired by HPT (Lirui Wang et al., 2024) and pi0 (Kevin Black et al., 2024).

    In essence, this class handles heterogeneities from two sources in two different ways:
    - embodiment heterogeneities: the heterogeneities in hardware is handled by switching between 'stems' and 'heads' (as in HPT)
    - task heterogeneities: the heterogeneities in tasks is handled by different prompts (as in pi0)

    If there are no heterogeneities, stems and trunks can be identity functions that does not accept any prompts and only
    the heads are used.

    TODO: load balancing between different modules. BIG TODO due to the complexity of the problem
    """

    module_registry: nn.ModuleDict = nn.ModuleDict()  # shared registry for all modules. this makes sharing modules easier
    module_info: ClassVar[Dict[str, tuple[str, int, int, List[str]]]] = {}  # module_name: (classname, num_params, num_trainable_params, used_in)
    module_batchsize: ClassVar[Dict[str, int]] = {}  # batchsize for each module

    """
    The advent of transformers has made it easy to handle highly heterogeneous data.
    This structure assumes that the trunk has the ability to handle any kind of data.
    """
    domain_registry: ClassVar[Dict[str, tuple[str, str, str]]] = {}  # registry holding the stem, trunk, and head for each domain

    logger = logging.getLogger("ActionModel")

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
        if "<unregistered>" in name:
            raise ValueError("<unregistered> is a reserved keyword. Please choose another name >:-(")
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
        self.module_info[name] = (module.__class__.__name__, num_params, num_trainable_params, [])

    def delete_module(self, name: str) -> None:
        """
        Delete the module from the action model's module registry.

        Args:
            name (str): the name of the module to delete
        """
        if name not in self.module_registry:
            raise ValueError(f"Module with name {name} is not registered.")
        del self.module_registry[name]
        del self.module_info[name]  # TODO: mark domains that use the deleted module as "<unregistered>"

    def print_module_info(self) -> None:
        """
        Print the information about the modules in the module registry.
        """
        table = prettytable.PrettyTable()
        table.field_names = ["Name", "Class", "Num Params", "Num Trainable Params", "Used In"]
        for name, info in self.module_info.items():
            table.add_row([name, *info])
        print(table)  # noqa: T201

    def register_domain(self, domain: str, stem: str, trunk: str, head: str) -> None:
        """
        Register the stem, trunk, and head for a domain.

        Args:
            domain (str): the domain to register
            stem (str): the stem to use
            trunk (str): the trunk to use
            head (str): the head to use

        Raises:
            ValueError: if the stem is not registered
            ValueError: if the trunk is not registered
            ValueError: if the head is not registered
            ValueError: if the domain is already registered
        """
        # for each of the components, check if they are registered
        if stem not in self.module_registry:
            raise ValueError(f"Stem {stem} is not registered.")
        if trunk not in self.module_registry:
            raise ValueError(f"Trunk {trunk} is not registered.")
        if head not in self.module_registry:
            raise ValueError(f"Head {head} is not registered.")

        # register the domain
        if domain in self.domain_registry:
            raise ValueError(f"Domain {domain} is already registered.")
        self.domain_registry[domain] = (stem, trunk, head)
        # update the module info
        self.module_info[stem][3].append(domain + ".stem")
        self.module_info[trunk][3].append(domain + ".trunk")
        self.module_info[head][3].append(domain + ".head")

    def delete_domain(self, domain: str) -> None:
        """
        Delete the domain from the domain registry.

        Args:
            domain (str): the domain to delete

        Raises:
            ValueError: if the domain is not registered
        """
        if domain not in self.domain_registry:
            raise ValueError(f"Domain {domain} is not registered.")
        stem, trunk, head = self.domain_registry[domain]
        del self.domain_registry[domain]
        # update the module info
        self.module_info[stem][3].remove(domain + ".stem")
        self.module_info[trunk][3].remove(domain + ".trunk")
        self.module_info[head][3].remove(domain + ".head")

    def print_domain_info(self) -> None:
        """
        Print the information about the domains in the domain registry.
        """
        table = prettytable.PrettyTable()
        table.field_names = ["Domain", "Stem", "Trunk", "Head"]
        for domain, (stem, trunk, head) in self.domain_registry.items():
            table.add_row([domain, stem, trunk, head])
        print(table)  # noqa: T201

    def configure_async_module_batchsize(self, module_batchsize: Dict[str, int]) -> None:
        """
        Configure the batchsize of the modules for running the model in async.

        Args:
            module_batchsize (Dict[str, int]): the batchsize for each module
        """

        for name, batchsize in module_batchsize.items():
            if name in self.module_registry:
                self.module_batchsize[name] = batchsize
            else:
                self.logger.warning(f"Module {name} does not exist in registry.")

    def run_async(self, inputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Run the different modules async on the inputs. Useful for training and evaluation on large batches.

        This method creates queues for each module and runs them in a producer-consumer pattern.
        Theoretically, this could even work on multiple GPUs.

        Args:
            inputs (Dict[str, Dict[str, torch.Tensor]]): the inputs to the model

        Returns:
            Dict[str, torch.Tensor]: the outputs of the model
        """
        # TODO: implement run_async
        raise NotImplementedError("run_async is not implemented")

    def run_for_domain(self, domain, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run the model for a specific domain. This ignores async batch sizes set.

        Args:
            domain (str): the domain to run the model for
            inputs (Dict[str, torch.Tensor]): the inputs to the model

        Returns:
            torch.Tensor: the output of the model
        """
        stem, trunk, head = self.domain_registry[domain]
        stem_module = self.module_registry[stem]
        trunk_module = self.module_registry[trunk]
        head_module = self.module_registry[head]

        # run the stem
        stem_output = stem_module(inputs)
        # run the trunk
        trunk_output = trunk_module(stem_output)
        # run the head
        head_output = head_module(trunk_output)

        return head_output


class CyberAgent:
    def __init__(self, model: ActionModel):
        self.model = model

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return self.model.get_action(obs)

    # def train(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray) -> None:
    #     self.model.train(obs, action, reward, next_obs)
