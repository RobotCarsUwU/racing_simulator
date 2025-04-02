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
    def computeCost():
        pass

    @abstractmethod
    def train():
        pass

    def relu(value):
        return np.max(0, value)
