from abc import ABC, abstractmethod
from basic_layers import Layer
import numpy as np

from initializer import Initializer
from loss import Loss
from optimizer import Optimizer


class Model(ABC):
    """only supports sequential list of layers -> if need paralell e.g. add, use respective layer"""
    def __init__(self, layers: list[Layer]):
        self.num_iterations = None
        self.layers = layers  # don't put loss in layers
        self.epochs = None
        self.initializer = None
        self.mini_batch_size = None
        self.optimizer = None
        self.loss_func = None

        # todo: support multiple initialization methods

    def fit(self, X, Y, epochs: int, initializer: Initializer, optimizer: Optimizer, loss_func: Loss, mini_batch_size: int = 64, num_iterations: int = 1000) -> None:
        # iterate through epoch



        # get input/output layer size

        self.epochs = epochs
        self.initializer = initializer
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.num_iterations = num_iterations
        
        data_size = X.shape[0]

        # initialize all trainable parameters randomly
        self._initialize_params()

        # train loop
        for epoch_idx in range(self.epochs):
            for iter_idx in range(self.num_iterations):
                # select mini-batch randomly
                selected_indices = self._get_mini_batch_indices(data_size)
                X_batch = X[selected_indices]
                Y_batch = Y[selected_indices]
    
    
                self._forward(X_batch, Y_batch)
                self._backward()
                self.optimizer.step()

    def predict(self, X_test):
        ...

    def evaluate(self, X_test, y_test):
        ...

    def _initialize_params(self):
        for layer in self.layers:
            layer.initialize_params(self.initializer)

    def _forward(self, X, y) -> np.ndarray:
        """returns loss"""

        for layer in self.layers:
            X = layer.forward(X)


        # calculate loss at the end using the output from the last layer and y
        loss = self.loss_func(X, y)
        return loss

    def _backward(self) -> np.ndaray:
        """for each layer call backward method"""

        # loss backward
        dX = self.loss_func.backward()

        for idx in range(len(self.layers)-1, -1, -1):
            dX = self.layers[idx].backward(dX)

        return dX

    def get_trainable_params(self):
        all_params = []
        for layer in self.layers:
            params = layer.get_params()
            all_params.extend(params)

        return all_params

    def get_gradients(self):
        all_gradients = []
        for layer in self.layers:
            grad = layer.get_param_gradients()
            all_gradients.append(grad)

        return all_gradients

    def _get_mini_batch_indices(self, data_size: int) -> np.array:
        return np.random.choice(data_size, self.mini_batch_size)
