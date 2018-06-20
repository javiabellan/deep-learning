import numpy as np
np.random.seed(1)
from functions import *


# ------ NETWORK ARCH ------
#
#  Layers = 1 (no hidden layer)
#  Epochs = 100
#  LR     = 1 (constant)
#  SGD    = BATCH
#  Loss   = MSE (with added 1/2)


# Hyper parameters
numer_of_epoch = 100
learning_rate  = 1


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
w1 = np.random.randn(3,1)


# Training
for epoch in range(numer_of_epoch):

    # Foward
    layer_1     = x.dot(w1)
    layer_1_act = sigmoid(layer_1)
    loss        = halfMSE(layer_1_act, y)

    print("Current Epoch : ", epoch, " current loss :", loss.sum())

    # Backward (SGD - BATCH)
    grad_1_part_1 = d_halfMSE(layer_1_act, y)
    grad_1_part_2 = d_sigmoid(layer_1)
    grad_1_part_3 = x
    grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

    # Weight Update
    w1 -=  learning_rate * grad_1


layer_1 = x.dot(w1)
layer_1_act = sigmoid(layer_1)

print("\n\nFinal : " ,layer_1_act[:,-1])
print("Final Round: " ,np.round(layer_1_act[:,-1]))
print("Ground Truth : ",y[:,-1])
print("W1 : ",w1[:,-1])



# ------ References -------
#
# Original code: https://gitlab.com/jae.duk.seo/Only_Numpy/blob/master/1_artificial_neural_networks_basic/1_constant_learning_rate.py
# Explanation: http://mccormickml.com/2014/03/04/gradient-descent-derivation/
#