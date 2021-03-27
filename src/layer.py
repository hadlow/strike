import numpy as np

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons).astype(np.float32) * np.sqrt(1. / n_neurons)
        self.biases = np.zeros((n_neurons))

    def forward(self, inputs):
        self.layer = np.dot(inputs, self.weights) + self.biases
        return self.layer

    def backward(self, error, a):
        self.d_w = (1. / n_samples) * np.matmul(error, a)
        self.d_b = (1. / n_samples) * np.sum(error, axis=1, keepdims=True)
        return self

class ReLU:
    def forward(self, inputs):
        self.layer = np.maximum(0, inputs)
        return self.layer

    def backward(self, inputs):
        return (self.layer > 0) * 1

class Sigmoid:
    def forward(self, inputs):
        self.layer = 1. / (1. + np.exp(-inputs))
        return self.layer

    def backward(self, inputs):
        return inputs - np.exp(self.layer) * inputs.sum(axis=1).reshape((-1, 1))

