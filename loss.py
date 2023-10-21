from abc import ABC, abstractmethod

from basic_layers import Layer
import numpy as np

class Loss(ABC):
    """similar to Layer, but only has forwad and backward"""
    @abstractmethod
    def forward(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        ...

    def backward(self) -> np.ndarray:
        ...


class SoftmaxAndCrossEntropy(Loss):
    def __init__(self) -> None:
        self.Y = None
        self.predicted_Y = None
        self.mean_entropy = None  # mean loss

    def forward(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
            X: output from ReLU. should have size of (batch_size, output_size)
            Y: one-hot encoded labels. should have size of (batch_size, output_size)

            Note: y doesn't need to be one-hot encoded
        """
        self.Y = Y

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

    def backward(self) -> np.array:
        batch_size = self.Y.shape[0]
        if self.predicted_Y.size == self.Y.size:
            dLdW = (self.predicted_Y - self.Y) / batch_size
        else:
            dLdW = self.predicted_Y.copy()
            dLdW[np.arange(batch_size), self.Y] -= 1
            dLdW = dLdW / batch_size

        return dLdW