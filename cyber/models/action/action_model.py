# action_model.py
# Description: This file contains the ActionModel class, which is responsible for handling the action model.

import numpy as np

from cyber.models import CyberModule


class ActionModel(CyberModule):
    """ActionModel class for learning-based policies.
    This class is the base class for all action models.

    The design of this class is inspired by HPT (Lirui Wang et al., 2024) and pi0 (Kevin Black et al., 2024).

    In essence, this class handles heteroneuity from two sources in two different ways:
    - embodiment heteroneuity: the heteroneuity in hardware is handled by switching between 'stems' and 'heads' (as in HPT)
    - task heteroneuity: the heteroneuity in tasks is handled by different prompts (as in pi0)

    If there are no heteroneuities, heads and trunks can be identity functions that does not accept any prompts and only
    the heads are used.

    TODO: make this class more flexible
    """


class CyberAgent:
    def __init__(self, model: ActionModel):
        self.model = model

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return self.model.get_action(obs)

    # def train(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray) -> None:
    #     self.model.train(obs, action, reward, next_obs)
