import numpy as np


# Activation function



################################################### Sigmoid
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1- sigmoid(x))

################################################### Tanh
def tanh(x):
	# return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return np.tanh(x)

def d_tanh(x):
	# return 1 - np.power(tanh(x), 2)
    return 1 - tanh(x) ** 2

################################################### Rectified Linear Unit
def relu(x):
	return max(0, x)

def d_relu(x):
	return 1 if x > 0 else 0

################################################### Leaky ReLU
def leakyrelu(x, alpha):
	return max(alpha * x, x)

def d_leakyrelu(x, alpha):
	return 1 if x > 0 else alpha

################################################### Exponential Linear Unit
def ELU(x);
	pass

################################################### Sofmax
def softmax():
	pass

################################################### Sofplus
def softplus():
	pass