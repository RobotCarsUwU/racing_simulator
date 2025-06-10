import numpy as np
import sys
import os

# sys.path.append(os.path.abspath(".."))
from ..NeuralNetwork import MyNeuralNetwork

class MLP(MyNeuralNetwork):

    def __init__(self, input_size, hidden_size, output_size, alpha):
        self._alpha = alpha
        self._input_size = input_size
        self._output_size = output_size

        self._layer_size = [input_size] + hidden_size + [output_size]

        self._weight = []
        self._bias = []
        for i in range(len(self._layer_size) - 1):
            W = (np.random.randn(self._layer_size[i], self._layer_size[i + 1])
                * np.sqrt(2.0 / self._layer_size[i]))
            b = np.zeros((1, self._layer_size[i + 1]))
            self._weight.append(W.astype(np.float32))
            self._bias.append(b.astype(np.float32))

    def propagateForward(self, X_value) -> np.float32:
        self._z = []
        self._a = [X_value.astype(np.float32)]

        for i in range(len(self._weight) - 1):
            z = np.dot(self._a[-1], self._weight[i]) + self._bias[i]
            self._z.append(z.astype(np.float32))
            a = self.relu(z.astype(np.float32))
            self._a.append(a.astype(np.float32))

        output_z = np.dot(self._a[-1], self._weight[-1]) + self._bias[-1]
        self._z.append(output_z.astype(np.float32))
        output = np.tanh(output_z.astype(np.float32))
        self._a.append(output.astype(np.float32))
        return output

    def propagateBackward(self, y_value, output):
        weight_gradient = [None] * len(self._weight)
        bias_gradient = [None] * len(self._bias)

        delta = self.derivateLoss(output, y_value).astype(np.float32)

        weight_gradient[-1] = np.dot(self._a[-2].T, delta).astype(np.float32)
        bias_gradient[-1] = np.sum(delta, axis=0, keepdims=True).astype(np.float32)

        for i in range(len(self._weight) - 2, - 1, - 1):
            delta = np.dot(delta, self._weight[i + 1].T) * self.derivateRelu(self._z[i]).astype(np.float32)
            weight_gradient[i] = np.dot(self._a[i].T, delta).astype(np.float32)
            bias_gradient[i] = np.sum(delta, axis=0, keepdims=True).astype(np.float32)
        return weight_gradient, bias_gradient

    def computeParams(self, weight_gradient, bias_gradient, l2_lambda=1e-4):
        for i in range(len(self._weight)):
            self._weight[i] -= self._alpha * (weight_gradient[i] + l2_lambda * self._weight[i])
            self._bias[i] -= self._alpha * bias_gradient[i]

    def computeLoss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def derivateLoss(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size

    def train(self, X_values, y_values, epochs, size):
        X_values = X_values.astype(np.float32)
        y_values = y_values.astype(np.float32)
        number_values = X_values.shape[0]

        for epoch in range(epochs):
            indices = np.arange(number_values)
            np.random.shuffle(indices)
            X_shuffled = X_values[indices]
            y_shuffled = y_values[indices]

            for start_idx in range(0, number_values, size):
                end_idx = min(start_idx + size, number_values)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                output = self.propagateForward(X_batch)
                loss = self.computeLoss(output, y_batch)
                weight_gradient, bias_gradient = self.propagateBackward(y_batch, output)
                self.computeParams(weight_gradient, bias_gradient)
