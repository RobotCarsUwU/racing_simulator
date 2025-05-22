from abc import ABC, abstractmethod
import numpy as np


class MyNeuralNetwork(ABC):

    @abstractmethod
    def propagateForward():
        pass

    @abstractmethod
    def propagateBackward():
        pass

    @abstractmethod
    def computeParams():
        pass

    @abstractmethod
    def train():
        pass

    @abstractmethod
    def computeLoss():
        pass

    @abstractmethod
    def derivateLoss(y_pred, y_true):
        pass

    def relu(self, value):
        return np.maximum(0, value)

    def derivateRelu(self, value):
        return (value > 0).astype(float)
