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
            self._weight.append(W)
            self._bias.append(b)

    def propagateForward(self, X_value):
        self._z = []
        self._a = [X_value]

        for i in range(len(self._weight) - 1):
            z = np.dot(self._a[-1], self._weight[i]) + self._bias[i]
            self._z.append(z)
            a = self.relu(z)
            self._a.append(a)

        output_z = np.dot(self._a[-1], self._weight[-1]) + self._bias[-1]
        self._z.append(output_z)
        output = np.tanh(output_z)
        self._a.append(output)
        return output

    def propagateBackward(self, y_value, output):
        weight_gradient = [None] * len(self._weight)
        bias_gradient = [None] * len(self._bias)

        delta = self.derivateLoss(output, y_value)

        weight_gradient[-1] = np.dot(self._a[-2].T, delta)
        bias_gradient[-1] = np.sum(delta, axis=0, keepdims=True)

        for i in range(len(self._weight) - 2, - 1, - 1):
            delta = np.dot(delta, self._weight[i + 1].T) * self.derivateRelu(self._z[i])
            weight_gradient[i] = np.dot(self._a[i].T, delta)
            bias_gradient[i] = np.sum(delta, axis=0, keepdims=True)
        return weight_gradient, bias_gradient

    def computeParams(self, weight_gradient, bias_gradient):
        for i in range(len(self._weight)):
            self._weight[i] -= self._alpha * weight_gradient[i]
            self._bias[i] -= self._alpha * bias_gradient[i]

    def computeLoss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def derivateLoss(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size

    def train(self, X_values, y_values, epochs, size):
        number_values = X_values.shape[0]

        for i in range(epochs):
            output = self.propagateForward(X_values)
            loss = self.computeLoss(output, y_values)
            weight_gradient, bias_gradient = self.propagateBackward(y_values, output)
            self.computeParams(weight_gradient, bias_gradient)
