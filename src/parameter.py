import numpy as np
from typing import Optional

class Parameter:
    def __init__(self, value: Optional[np.ndarray] = None, gradient: Optional[np.ndarray] = None, name: Optional[str] = None):
        self.value = value
        self.gradient = gradient
        self.name = name

    def reset_grad_to_zeroes(self):
        self.gradient = np.zeros_like(self.value)

