from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    def __init__(self, params: list):
        """
        the params must be pointers to numpy arrays not an integer; if integer, the parameter will not be updated globally
        :param params:
        """
        self.params = params

    def step(self, gradients: list) -> None:
        ...


class SGD(Optimizer):
    def __init__(self, params: list, learning_rate: float):
        super().__init__(params)
        self.learning_rate = learning_rate

    def step(self, gradients: list) -> None:
        """uses the gradient of each parameter to update its value, mutates the parameter"""

        for i in range(len(self.params)):
            self.params[i][:] = self.params[i] - self.learning_rate*gradients[i]

