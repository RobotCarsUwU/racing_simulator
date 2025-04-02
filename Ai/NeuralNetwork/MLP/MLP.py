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
