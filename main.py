import numpy as np
from typing import Union, Optional
from basic_layers import Linear, ReLU, SoftmaxAndCrossEntropy
from preprocessing import get_normalized_data

# one optimization is to flatten all the parameters (don't separate them)

class TwoLayerNeuralNetwork:
    def __init__(self, eta, epsilon, hidden_size: int = 3, mini_batch_size: Optional[int] = None,
                 do_track=False, weight_init_std = 0.01) -> None:
        # initialize mini_batch_size
        self.hidden_size = hidden_size
        self.eta = eta
        self.epsilon = epsilon
        self.mini_batch_size = mini_batch_size
        self.weight_init_std = weight_init_std
        self.do_track = do_track
        self.min_iters = 1000

        # initialize layers (instantiate Affine and SoftmaxAndCrossEntropy)
        affine1 = Linear()  # 0
        relu1 = ReLU()  # 1
        affine2 = Linear()  # 2
        softmax_and_loss = SoftmaxAndCrossEntropy()  # 3

        self.layers = [affine1, relu1, affine2, softmax_and_loss]

        # initialize placeholders for the optimum weights/biases
        self.opt_params = {}

    def fit(self, X: np.array, Y: np.array) -> None:
        """
            X: a matrix representing features with some predictors
            Y: a matrix representing one-hot encoded labels

            Note: inputs must not contain any strings
            Note: assumes that all rows have the same length; if inputting image, it should be flattened
            Note: labels must start from 0 and be integers; labels shouldn't be sparse; labels must be one-hot encoded
        """
        # get input/output layer size
        data_size = X.shape[0]
        input_size = X.shape[1]
        output_size = max(np.argmax(Y, axis=1))+1  # assumes that there is 0; if given label is one-hot encoded

        # check if the labels are one-hot encoded

        # initialize parameters (weights/biases) and do one iteration of forward propagation
        current_W1 = self.weight_init_std*np.random.randn(input_size, self.hidden_size)
        current_B1 = np.zeros(self.hidden_size)
        self.layers[0].set_params(current_W1, current_B1)

        current_W2 = self.weight_init_std*np.random.randn(self.hidden_size, output_size)
        current_B2 = np.zeros(output_size)
        self.layers[2].set_params(current_W2, current_B2)

        # current_params = np.stack([W1, B1, W2, B2], axis=0)

        # SGD
        i = 0
        while True:
            # get random mini batch and do forward propagation

            selected_indices = self._get_mini_batch_indices(data_size)
            labels = Y[selected_indices]
            self.layers[-1].set_correct_labels(labels)

            inp = X[selected_indices]
            for layer in self.layers:
                inp = layer.forward(inp)

            dLdW1, dLdB1, dLdW2, dLdB2 = self._get_grad()

            # current_params = current_params - self.eta * grad

            # dLdW1, dLdB1, dLdW2, dLdB2
            current_W1 = current_W1 - self.eta * dLdW1
            current_B1 = current_B1 - self.eta * dLdB1
            current_W2 = current_W2 - self.eta * dLdW2
            current_B2 = current_B2 - self.eta * dLdB2

            # update parameters inside the layers
            self.layers[0].set_params(current_W1, current_B1)
            self.layers[2].set_params(current_W2, current_B2)

            if self._get_magnitude_of_gradient([dLdW1, dLdB1, dLdW2, dLdB2]) > self.epsilon and i > self.min_iters:
                break

            i += 1

            # add tracking tool here
            if self.do_track and i % 200 == 0:
                print("current magnitude of gradient: ", self._get_magnitude_of_gradient([dLdW1, dLdB1, dLdW2, dLdB2]))

        # W1, B1 = current_params[0], current_params[1]
        # W2, B2 = current_params[2], current_params[3]
        self.opt_params.update({"W1": current_W1, "B1": current_B1, "W2": current_W2, "B2": current_B2})

    def predict(self, X: np.array, params=None) -> np.array:

        if params is not None:
            # when using previously trained parameters to predict
            pass

        # no need for softmax and loss calculation; just choose the max output for each row
        inp = X
        for idx in range(len(self.layers) - 1):
            inp = layer.forward(inp)

        predictions = np.argmax(inp, axis=1)

        return predictions

    def _get_grad(self) -> np.array:
        # do backpropagation here

        dinp = 1
        for idx in range(len(self.layers)-1, -1, -1):
            dinp = self.layers[idx].backward(dinp)

        # collect all partial derivates and return them
        dLdW1, dLdB1 = self.layers[0].get_updated_params()
        dLdW2, dLdB2 = self.layers[2].get_updated_params()

        # grad = np.stack([dLdW1, dLdB1, dLdW2, dLdB2], axis=0)

        return dLdW1, dLdB1, dLdW2, dLdB2

    def _get_mini_batch_indices(self, data_size: int) -> np.array:
        return np.random.choice(data_size, self.mini_batch_size)

    def _get_magnitude_of_gradient(self, grad) -> np.array:
        sum_ = 0
        for pdts in grad:
            sum_ += np.sum(pdts**2)

        return sum_**(1/2)

X_train, Y_train, X_test, Y_test = get_normalized_data()
network = TwoLayerNeuralNetwork(0.1, 0.2, mini_batch_size=100, hidden_size=50, do_track=True)
network.fit(X_train, Y_train)
print(network.opt_params)