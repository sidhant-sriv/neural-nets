import numpy as np
class Activation:
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    def relu(self, n):
        return np.maximum(0,n)
    def tanh(self, x):
        return np.tanh(x)
