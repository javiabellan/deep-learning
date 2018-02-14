import numpy as np
np.random.seed(1234)


# ------ NETWORK ARCH ------
#
#  Layers = 2 (1 hidden layer)
#  Epochs = 300
#  LR     = 
#  SGD    = BATCH
#  Loss   = MSE (with added 1/2)


# Hyper parameters
numer_of_epoch = 300
learning_rate_1  = 0.01
learning_rate_2  = 0.1


# Data
x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

y = np.array([
    [0],
    [0],
    [0],
    [1]
])

# Variables
w1 = np.random.randn(3,5)
w2 = np.random.randn(5,1)

for iter in range(numer_of_epoch):

    # Foward
    layer_1     = x.dot(w1)
    layer_1_act = tanh(layer_1)

    layer_2     = layer_1_act.dot(w2)
    layer_2_act = tanh(layer_2)

    cost        = MSE(layer_2_act, y)


    # Backward
    grad_2_part_1 = d_MSE(layer_2_act, y)
    grad_2_part_2 = d_tanh(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2) 

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_tanh(layer_1)
    grad_1_part_3 = x
    grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)   

    w1 -= learning_rate_1*grad_1
    w2 -= learning_rate_2*grad_2
    


layer_1     = x.dot(w1)
layer_1_act = tanh(layer_1)
layer_2     = layer_1_act.dot(w2)
layer_2_act = tanh(layer_2)

print(layer_2_act)






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
