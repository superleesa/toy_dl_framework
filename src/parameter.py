import numpy as np

class Parameter:
    def __init__(self, value: np.ndarray = None, gradient: np.ndarray = None, name: str = None):
        self.value = value
        self.gradient = gradient
        self.name = name

    def reset_grad_to_zeroes(self):
        self.gradient = np.zeroes(self.gradient.shape)

