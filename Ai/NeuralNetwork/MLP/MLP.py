import numpy as np
import sys
import os

from NeuralNetwork import MyNeuralNetwork
sys.path.append(os.path.abspath(".."))


class MLP(MyNeuralNetwork):

    def __init__(self, input_size, hidden_size, output_size, alpha):
        self._alpha = alpha
        self._input_size = input_size
        self._output_size = output_size

        self._layer_size = [input_size] + hidden_size + [output_size]

        for i in range(len(self._layer_size) - 1):
            W = (np.random.randn(self._layer_size[i], self._layer_size[i + 1])
                * np.sqrt(2.0 / self._layer_size[i]))
            b = np.zeros((1, self._layer_size[i + 1]))
            self._weight.append(W)
            self._bias.append(b)

    def propagateForward(self, X_value):
        self._z = []
        self._a = [X_value]

        for i in range(len(self._weights) - 1):
            z = np.dot(self._a[-1], self.weight[-1]) + self.biases[-1]
            self._z.append(z)
            a = self.relu(z)
            self._a.append(a)

        output = np.dot(self._a[-1], self._weights[-1]) + self._bias[-1]
        self._z.append(output)
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
            self._weight[i] -= self._alpha * weight_gradient
            self._bias[i] -= self._alpha * bias_gradient

    def computeLoss(y_pred, y_true):
        return np.mean((y_pred, y_true) ** 2)

    def derivateLoss(y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size
