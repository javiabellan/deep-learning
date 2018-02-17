import numpy as np


# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp( -1 * x))

def d_sigmoid(x):
    return sigmoid(x) * (1- sigmoid(x))

# Activation function: tanh
def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - tanh(x) ** 2


# Loss function: Mean Squared Error
def MSE(pred, y):
    # return np.square(pred - y) / (len(pred))
    return np.square(pred - y).sum() / len(x)

def d_MSE(pred, y):
    # return (pred - y) / (len(pred) * 2)
    return (2/len(x)) * (pred - y)

# Loss function: One Half Mean Squared Error
def halfMSE(pred, y):
    return np.square(pred - y) / (len(pred) * 2)

def d_halfMSE(pred, y):
    return (pred - y) / len(pred)