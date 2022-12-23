# Implement XOR gate using a feed-forward neural network 
import numpy as np
#Input and Output
np.random.seed(0)
x = np.array([[0,0],[0,1],[1,0],[1,1], [1,0], [0,1], [1,1] , [0,0]])
y = np.array([[0],[1],[1],[0], [1] , [1] , [0] , [0]])

#Network Architecture
input_layer_n = 2
hidden_layer_n = 3
output_n = 1

# 2 3 1
hidden_weights = np.random.uniform(size= (input_layer_n, hidden_layer_n))
hidden_bias = np.random.uniform(size=(1, hidden_layer_n))

output_weights = np.random.uniform(size=(hidden_layer_n, output_n))
output_bias = np.random.uniform(size=(1, output_n))
#Activation
def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
#Hyper parameter
learning_rate = 0.1
epochs = 100000

# print(hidden_weights)
# print(hidden_bias)

# print(output_weights)
# print(output_bias)

for i in range(epochs):
    #Forward propagation
    hidden_layer_activation = np.dot(x, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_activation)
    # Backpropagation
    error = y - predicted_output
    d_predicted_output = error*(sigmoid_derivative(predicted_output))

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * learning_rate
    hidden_weights += x.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * learning_rate
print(x)
print("Hidden Weights")
print(hidden_weights)
print("Hidden Bias")
print(hidden_bias)
print("Output weights")
print(output_weights)
print("Output bias")
print(output_bias)
print("Predicted Output")
print(predicted_output)
print(y)
print(sum(error*error)*100/len(error))
