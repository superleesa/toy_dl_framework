import numpy as np
from typing import Union, Optional
from abc import ABC, abstractmethod

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


class SoftmaxAndCrossEntropy(Layer):
    def __init__(self) -> None:
        self.Y = None
        self.predicted_Y = None
        self.mean_entropy = None  # mean loss

    def set_correct_labels(self, Y: np.ndarray) -> None:
        self.Y = Y

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
            X: output from ReLU. should have size of (batch_size, output_size)
            Y: one-hot encoded labels. should have size of (batch_size, output_size)

            Note: y doesn't need to be one-hot encoded
        """

        # softmax
        X = X - np.max(X, axis=-1, keepdims=True)  # adding a constant to avoid overflow
        self.predicted_Y = np.exp(X) / np.sum(np.exp(X), axis=-1, keepdims=True)

        # cross entropy (= lack of order => smaller the better)
        if self.predicted_Y.ndim == 1:
            self.predicted_Y = self.predicted_Y.reshape(1, self.predicted_Y.size)
            self.Y = self.Y.reshape(1, self.Y.size)

        # check if Y is one-hot vector or not
        if self.Y.shape == self.predicted_Y.shape:
            self.Y = np.argmax(self.Y, axis=1)

        batch_size = self.Y.shape[0]
        delta = 1e-7
        self.mean_entropy = -np.sum(np.log(self.predicted_Y[np.arange(batch_size), self.Y] + delta)) / batch_size
        # print(self.mean_entropy)
        return self.mean_entropy

    def backward(self, dinpt: np.ndarray = 1) -> np.array:
        batch_size = self.Y.shape[0]
        if self.predicted_Y.size == self.Y.size:
            dLdW = (self.predicted_Y - self.Y) / batch_size
        else:
            dLdW = self.predicted_Y.copy()
            dLdW[np.arange(batch_size), self.Y] -= 1
            dLdW = dLdW / batch_size

        return dLdW


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


class Dense(Layer):
    def __init__(self):
        self.X = None
        self.W = None
        self.B = None
        self.dLdW = None
        self.dLdB = None

    def set_params(self, W: np.ndarray, B: np.ndarray):
        self.W = W
        self.B = B

    def get_updated_params(self):
        return self.dLdW, self.dLdB

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
            X: should have size of (batch_size, input_size) or (batch_size, hidden_size)
            W: should have size of (input_size, hidden_size) or (hidden_size, output_size)
            B: should have size of (batch_size, hidden_size)

            Note: all of these are matrices not vectors, because it consideres batch inputs
        """
        self.X = X
        X2 = np.dot(X, self.W) + self.B

        return X2

    def backward(self, d_inpt: np.ndarray) -> np.ndarray:
        # returns derivatives for inputs, weights and biases
        # if this affine layer is the first one, then the derivatives for inputs should be discarded

        dLdX = np.dot(d_inpt, self.W.T)
        self.dLdW = np.dot(self.X.T, d_inpt)
        self.dLdB = np.sum(d_inpt, axis=0)

        return dLdX