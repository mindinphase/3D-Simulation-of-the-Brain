import numpy as np

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * 0.1
        self.bias = 0.0
        self.activity = 0.0
        self.learning_rate = 0.01

    def activate(self, inputs):
        self.activity = np.tanh(np.dot(self.weights, inputs) + self.bias)
        return self.activity

    def hebbian_update(self, inputs):
        delta_w = self.learning_rate * self.activity * inputs
        self.weights += delta_w
        self.bias += self.learning_rate * self.activity