from abc import ABC, abstractmethod
from layers import Layer, Softmax
import numpy as np

from initializer import Initializer
from loss import Loss, CrossEntropy, SoftmaxThenCrossEntropy
from optimizer import Optimizer
from metric import Metric


class Model(ABC):
    """only supports sequential list of layers -> if need paralell e.g. add, use respective layer"""
    def __init__(self, layers: list[Layer]):
        self.layers = layers  # don't put loss in layers
        self.epochs = None
        self.initializer = None
        self.mini_batch_size = None
        self.optimizer = None
        self.loss_func = None

        self.removed_softmax = False

        # todo: support multiple initialization methods

    def fit(self, X, Y, loss_func: Loss, optimizer: Optimizer, epochs: int = 3, mini_batch_size: int = 64) -> None:
        """fit API similar to keras that automatically does everything, without writing a training loop"""

        # get input/output layer size
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer
        self.loss_func = loss_func

        # check if we can combine softmax & cross-entropy for stability
        if isinstance(loss_func, CrossEntropy) and isinstance(self.layers[-1], Softmax):
            self.layers.pop()
            self.loss_func = SoftmaxThenCrossEntropy()
            self.removed_softmax = True
        
        batch_size = X.shape[0]
        num_iterations_per_epoch = int(batch_size / 64)

        # initialize all trainable parameters randomly
        self._initialize_params()

        # train loop
        for epoch_idx in range(self.epochs):
            for iter_idx in range(num_iterations_per_epoch):
                # select mini-batch randomly
                selected_indices = self._get_mini_batch_indices(batch_size)
                X_batch = X[selected_indices]
                Y_batch = Y[selected_indices]

                # print("===============")
                # print([param.value[0] for param in self.optimizer.params])
                # if iter_idx != 0:
                #     print([param.gradient[0] for param in self.optimizer.params])
    
    
                self.forward_then_loss(X_batch, Y_batch)
                self.backward()
                self.optimizer.step()

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)

        if self.removed_softmax:
            softmax = Softmax()
            X = softmax.forward(X)

        return X

    def evaluate(self, X_test, y_test, metric: Metric):
        y_pred = self.predict(X_test)
        score = metric.get_score(y_test, y_pred)
        return score

    def _initialize_params(self):
        for layer in self.layers:
            layer.initialize_params()

    def forward_then_loss(self, X, y) -> np.ndarray:
        """returns loss"""

        for layer in self.layers:
            X = layer.forward(X)


        # calculate loss at the end using the output from the last layer and y
        loss = self.loss_func.forward(X, y)
        return loss

    def backward(self) -> None:
        """for each layer call backward method"""

        # loss backward
        dX = self.loss_func.backward()

        for idx in range(len(self.layers)-1, -1, -1):
            dX = self.layers[idx].backward(dX)

    def get_trainable_params(self):
        all_params = []
        for layer in self.layers:
            params = layer.get_params()
            all_params.extend(params)

        return all_params


    def _get_mini_batch_indices(self, data_size: int) -> np.array:
        return np.random.choice(data_size, self.mini_batch_size)
