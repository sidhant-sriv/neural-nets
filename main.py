# import numpy as np
# np.random.seed(0)
# X = [[1,2,3,4],[4,3,2,1],[2,4,6,8],[8,6,4,2],[2,3,4,5]]
# Y = [1,0,1,0]
# def sigmoid(x):
#         return 1/(1 + np.exp(-x)) 
# class Layer:
#     def __init__(self, input_dims, neurons):
#         self.weights = 0.10 * np.random.randn(input_dims, neurons)
#         self.bias = 0.10*np.random.random((1,neurons))
#     def forward(self, prev_activation):
#         self.output = sigmoid(np.dot(prev_activation, self.weights) + self.bias)
# class Model:
#     pass
# layer1 = Layer(4,5)
# layer2 = Layer(5,2)
# layer1.forward(X)
# layer2.forward(layer1.output)
# print(layer2.output)
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X = [[1, 1, 1, 1], [-1, -1, -1, -1], [1, -1, 1, -1], [-1, 1, -1, 1], [-1, -1, 1, 1]]
Y = [0.25, 0.95, 0.25, 0.25, 0.95]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Layer:
    def __init__(self, input_dims, neurons):
        self.weights = np.array([-0.27, -0.43, 0.33, 0.23])
        self.bias = np.array([2.3])

    def forward(self, prev_activation):
        self.output = sigmoid(np.dot(prev_activation, self.weights) + self.bias)
layer1 = Layer(4,5)
layer1.forward(X)
print(layer1.output - np.array(Y))
res = []
for _ in range(100):
    layer1.forward(X)
    E = layer1.output - np.array(Y)
    print(np.dot(E, E.T))
    res.append(np.dot(E,E.T))
    plt.plot(res)
    plt.pause(0.1)
plt.draw()
