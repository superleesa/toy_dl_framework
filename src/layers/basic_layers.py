import numpy as np
from typing import Union, Optional
from abc import ABC, abstractmethod

from initializer import Initializer, ZerosInitializer, NormalInitializer
from parameter import Parameter

class Layer(ABC):
    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self, dX: np.ndarray) -> np.ndarray:
        """
        IMPORTANT: mutates gradients of trainable parameters in the layer

        :param dX: the derivative flowed from the parent layer
        :return:
        """
        ...

    def initialize_params(self):
        """layers without parameters does nothing"""
        return

    def get_params(self) -> list:
        return []


class ReLU(Layer):

    def __init__(self) -> None:
        self.is_activated = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
            X: should have size of batch_size*hidden_layer
        """
        self.is_activated = X <= 0
        out = X.copy()  # don't mutate x
        out[self.is_activated] = 0
        return out

    def backward(self, d_inpt: np.ndarray) -> np.ndarray:
        """
            derivative_inpt: should have size of batch_size*hidden_layer (same shape as X)
        """
        d_inpt[self.is_activated] = 0
        return d_inpt


class Softmax(Layer):

    def forward(self, X: np.ndarray) -> np.ndarray:
        num_data = len(X)
        exp = np.exp(X)
        return exp / np.sum(exp, axis=1).reshape((num_data, 1))

    def backward(self, dX: np.ndarray) -> np.ndarray:
        # todo
        pass


class Linear(Layer):

    def __init__(self, in_features, out_features, weight_initializer: Initializer = None, bias_initializer: Initializer = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # input
        self.X = None

        # params
        self.W = Parameter()
        self.b = Parameter()

        # bias initializer
        self.weight_initializer = weight_initializer if weight_initializer is not None else NormalInitializer()
        self.bias_initializer = bias_initializer if bias_initializer is not None else ZerosInitializer()

    def initialize_params(self):
        self.W.value = self.weight_initializer.initialize_array([self.in_features, self.out_features])
        self.b.value = self.bias_initializer.initialize_array([self.out_features])

    def get_params(self) -> list:
        return [self.W, self.b]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
            X: should have size of (batch_size, input_size) or (batch_size, hidden_size)
            W: should have size of (input_size, hidden_size) or (hidden_size, output_size)
            B: should have size of (batch_size, hidden_size)

            Note: all of these are matrices not vectors, because it consideres batch inputs
        """
        self.X = X
        W = self.W.value
        b = self.b.value

        X2 = np.dot(X, W) + b

        return X2

    def backward(self, d_inpt: np.ndarray) -> np.ndarray:
        # returns derivatives for inputs, weights and biases
        # if this affine layer is the first one, then the derivatives for inputs should be discarded
        W = self.W.value

        dLdX = np.dot(d_inpt, W.T)
        self.W.gradient = np.dot(self.X.T, d_inpt)
        self.b.gradient = np.sum(d_inpt, axis=0)

        return dLdX


class Embedding(Layer):
    def __init__(self, vocab_size: int, embed_dim: int, embedding_initializer: Initializer = None) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_dim

        self.embedding = Parameter()
        self.embedding_initializer = embedding_initializer if embedding_initializer is not None else NormalInitializer()

        self.passed_vocab_ids = None

    def initialize_params(self):
        self.embedding.value = self.embedding_initializer.initialize_array([self.vocab_size, self.embed_size])

    def get_params(self) -> list:
        return [self.embedding]

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        # token_ids: should have size of (batch_size, max_seq_len)
        # gather
        self.passed_vocab_ids = token_ids
        return np.take(self.embedding.value, token_ids, axis=0)

    def backward(self, d_inpt: np.ndarray) -> np.ndarray:
        # d_inpt: should have size of (batch_size, max_seq_len, embed_size)

        self.embedding.reset_grad_to_zeroes()
        self.embedding.gradient[self.passed_vocab_ids.flatten()] += d_inpt.reshape(-1, self.embed_size)

        return self.embedding.gradient
