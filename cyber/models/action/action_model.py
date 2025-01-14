# action_model.py
# Description: This file contains the ActionModel class, which is responsible for handling the action model.

import numpy as np

from cyber.models import CyberModule


class ActionModel(CyberModule):
    """ActionModel class for learning-based policies.
    This class is the base class for all action models.
    It is desgisned for real-time inferencing on robotics applications.
    """

    def __init__(self, backbone: str, objective: str):
        super().__init__()


class CyberAgent:
    def __init__(self, model: ActionModel):
        self.model = model

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return self.model.get_action(obs)

    # def train(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray) -> None:
    #     self.model.train(obs, action, reward, next_obs)
