from abc import ABC, abstractmethod

import numpy as np


class Initializer(ABC):
    @abstractmethod
    def initialize_array(self, shape: list) -> np.ndarray:
        ...

class RandomInitializer(Initializer):
    def initialize_array(self, shape: list, std: int = 0.01) -> np.ndarray:
        return std*np.random.randn(*shape)

class ZerosInitializer(Initializer):
    def initialize_array(self, shape: list) -> np.ndarray:
        return np.zeros(shape)
