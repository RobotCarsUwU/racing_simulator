import numpy as np
import sys
import os

import tensorflow as tf

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
            W = tf.random.normal(
                [self._layer_size[i], self._layer_size[i + 1]]
            ) * tf.sqrt(2.0 / self._layer_size[i])
            b = tf.zeros([1, self._layer_size[i + 1]])
            self._weight.append(tf.Variable(tf.cast(W, tf.float32)))
            self._bias.append(tf.Variable(tf.cast(b, tf.float32)))

    def propagateForward(self, X_value):
        self._z = []
        self._a = [tf.cast(X_value, tf.float32)]

        for i in range(len(self._weight) - 1):
            z = tf.matmul(self._a[-1], self._weight[i]) + self._bias[i]
            self._z.append(tf.cast(z, tf.float32))
            a = self.relu(tf.cast(z, tf.float32))
            self._a.append(tf.cast(a, tf.float32))

        output_z = tf.matmul(self._a[-1], self._weight[-1]) + self._bias[-1]
        self._z.append(tf.cast(output_z, tf.float32))
        output = tf.tanh(tf.cast(output_z, tf.float32))
        self._a.append(tf.cast(output, tf.float32))
        return output

    def propagateBackward(self, y_value, output):
        weight_gradient = [None] * len(self._weight)
        bias_gradient = [None] * len(self._bias)

        delta = tf.cast(self.derivateLoss(output, y_value), tf.float32)

        weight_gradient[-1] = tf.cast(
            tf.matmul(tf.transpose(self._a[-2]), delta), tf.float32
        )
        bias_gradient[-1] = tf.cast(
            tf.reduce_sum(delta, axis=0, keepdims=True), tf.float32
        )

        for i in range(len(self._weight) - 2, -1, -1):
            delta = tf.matmul(delta, tf.transpose(self._weight[i + 1])) * tf.cast(
                self.derivateRelu(self._z[i]), tf.float32
            )
            weight_gradient[i] = tf.cast(
                tf.matmul(tf.transpose(self._a[i]), delta), tf.float32
            )
            bias_gradient[i] = tf.cast(
                tf.reduce_sum(delta, axis=0, keepdims=True), tf.float32
            )
        return weight_gradient, bias_gradient

    def computeParams(self, weight_gradient, bias_gradient, l2_lambda=1e-4):
        for i in range(len(self._weight)):
            self._weight[i].assign_sub(self._alpha * (
                weight_gradient[i] + l2_lambda * self._weight[i]
            ))
            self._bias[i].assign_sub(self._alpha * bias_gradient[i])

    def computeLoss(self, y_pred, y_true):
        return tf.reduce_mean(tf.square(y_pred - y_true))

    def derivateLoss(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / tf.cast(tf.size(y_true), tf.float32)

    def train(self, X_values, y_values, epochs, size):
        X_values = tf.cast(X_values, tf.float32)
        y_values = tf.cast(y_values, tf.float32)
        number_values = tf.shape(X_values)[0]

        for epoch in range(epochs):
            indices = tf.random.shuffle(tf.range(number_values))
            X_shuffled = tf.gather(X_values, indices)
            y_shuffled = tf.gather(y_values, indices)

            for start_idx in range(0, number_values, size):
                end_idx = min(start_idx + size, number_values)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                output = self.propagateForward(X_batch)
                loss = self.computeLoss(output, y_batch)
                weight_gradient, bias_gradient = self.propagateBackward(y_batch, output)
                self.computeParams(weight_gradient, bias_gradient)
