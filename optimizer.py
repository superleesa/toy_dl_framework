from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, params):
        """
        the params must be pointers to numpy arrays not an integer; if integer, the parameter will not be updated globally
        :param params:
        """
        self.params = params

    def step(self):
        ...


class SGD(Optimizer):
    def __init__(self, params):
        super().__init__(params)

    def step(self):
        """uses the gradient of each parameter to update its value"""
