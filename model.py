from abc import ABC, abstractmethod
from basic_layers import Layer
import numpy as np

class Model(ABC):
    def __init__(self, layers: list[Layer], epochs: int, optimizer, loss_func):
        self.layers = layers
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_func = loss_func

        # todo: support multiple initialization methods

    def fit(self, optimizer):
        # iterate through epoch


        # choose data randomly

        for epoch_idx in range(self.epochs):
            self.step()
        ...

    def predict(self, X_test):
        ...

    def evaluate(self, X_test, y_test):
        ...

    def _forward(self, X, y) -> np.ndarray:

        for layer in self.layers:
            X = layer.forward(X)


        # calculate loss at the end using the output from the last layer and y
        pass

    def _backward(self) -> np.ndaray:
        """for each layer call backward method"""

        dX = 1

        for idx in range(len(self.layers)-1, -1, -1):
            dX = self.layers[idx].backward(dX)

        return dX

    def step(self, X, y):
        self._forward(X, y)
        self._backward()
        self.optimizer.step()