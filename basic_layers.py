import numpy as np
from typing import Union, Optional
from abc import ABC, abstractmethod

from initializer import Initializer


class Layer(ABC):

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self, dX: np.ndarray) -> np.ndarray:
        """

        :param dX: the derivative flowed from the parent layer
        :return:
        """
        ...

    @abstractmethod
    def initialize_params(self, initializer: Initializer):
        ...

    @abstractmethod
    def get_params(self) -> list:
        ...

    @abstractmethod
    def get_param_gradients(self) -> list:
        ...


class ReLU(Layer):
    def __init__(self) -> None:
        self.is_zero = None

    def forward(self, X: np.ndarray) -> None:
        """
            X: should have size of batch_size*hidden_layer
        """
        self.is_zero = X <= 0
        out = X.copy()  # don't mutate x
        out[self.is_zero] = 0
        return out

    def backward(self, d_inpt: np.ndarray) -> np.ndarray:
        """
            derivative_inpt: should have size of batch_size*hidden_layer (same shape as X)
        """
        d_inpt[self.is_zero] = 0
        return d_inpt


class Linear(Layer):

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # input
        self.X = None

        # params
        self.W = None
        self.b = None

        # weights
        self.dLdW = None
        self.dLdB = None

    # def set_params(self, W: np.ndarray, b: np.ndarray):
    #     self.W = W
    #     self.b = b

    # def get_updated_params(self):
    #     return self.dLdW, self.dLdB

    def initialize_params(self, initializer):
        self.W = initializer.initialize_array([self.in_features, self.out_features])
        self.b = initializer.initialize_array([self.out_features])

    def get_params(self) -> list:
        return [self.X. self.b]

    def get_param_gradients(self) -> list:
        return [self.dLdW, self.dLdB]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
            X: should have size of (batch_size, input_size) or (batch_size, hidden_size)
            W: should have size of (input_size, hidden_size) or (hidden_size, output_size)
            B: should have size of (batch_size, hidden_size)

            Note: all of these are matrices not vectors, because it consideres batch inputs
        """
        self.X = X
        X2 = np.dot(X, self.W) + self.b

        return X2

    def backward(self, d_inpt: np.ndarray) -> np.ndarray:
        # returns derivatives for inputs, weights and biases
        # if this affine layer is the first one, then the derivatives for inputs should be discarded

        dLdX = np.dot(d_inpt, self.W.T)
        self.dLdW = np.dot(self.X.T, d_inpt)
        self.dLdB = np.sum(d_inpt, axis=0)

        return dLdX