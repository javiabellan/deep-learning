import numpy as np
np.random.seed(1234)


# ------ NETWORK ARCH ------
#
#  Layers = 3 (2 hidden layer)
#  Epochs = 300
#  LR     = 
#  SGD    = BATCH
#  Loss   = MSE (with added 1/2)


# Hyper parameters
numer_of_epoch = 150
learning_rate  = 0.0001

# Data
x = np.array([
    [1,2,1],
    [3,4,1],
    [5,6,1],
    [7,8,1]
])

y = np.array([
    [8],
    [16],
    [24],
    [32]
])


# Variables
w1 = np.random.randn(3,4)
w2 = np.random.randn(4,10)
w3 = np.random.randn(10,1)


for iter in range(num_epoch):
    
    layer_1 = x.dot(w1)
    layer_1_act = identity(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = identity(layer_2)
    
    layer_3 = layer_2_act.dot(w3)
    layer_3_act = identity(layer_3)

    cost = np.square(layer_3_act - y).sum() * 0.5
    print("Current Iter: ",iter, " current Cost: ",cost)

    grad_3_part_1= layer_3_act - y
    grad_3_part_2= d_identity(layer_3)
    grad_3_part_3=layer_2_act
    grad_3 =   grad_3_part_3.T.dot(grad_3_part_1*grad_3_part_2)
    
    grad_2_part_1= (grad_3_part_1 * grad_3_part_2).dot(w3.T)
    grad_2_part_2= d_identity(layer_2)
    grad_2_part_3=layer_1_act
    grad_2 =   grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2) 

    grad_1_part_1= (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2= d_identity(layer_1)
    grad_1_part_3= x
    grad_1 =     grad_1_part_3.T.dot(grad_1_part_1*grad_1_part_2)   

    w1 = w1 - lr*grad_1
    w2 = w2 - lr*grad_2
    w3 = w3 - lr*grad_3

print("----------------")    
print("After 100 Iter Result")
layer_1 = x.dot(w1)
layer_1_act = identity(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = identity(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = identity(layer_3)
print(layer_3_act)

print("----------------")    
print("Results Rounded: ")
print(np.round(layer_3_act))

print("----------------")    
print("Ground Truth: ")
print(y)

print("----------------")    
print("One Linear Equation ")
print("k = w1.dot(w2.dot(w3))")
k = w1.dot(w2.dot(w3))
one_liner = x.dot(k)
print(one_liner)

print("----------------")    
print("One Linear Equation Rounded: ")
print(np.round(one_liner))



# print("----------------")    
# print("Calculated Weigths: ")
# print(w1)
# print(w2)
# print(w3)



################################################### Activation functions


def identity(x):
      return x
  
def d_identity(x):
  return 1

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

################################################### FUNCIONES DE ERROR

# Least Absolute Deviations (LAD) = absolute error = L1-norm loss






# Least Squares Error (LSE) = Error cuadrático = L2-norm loss




# Mean Squared Error (MSE)
def MSE(pred, y):
    # return np.square(pred - y) / (len(pred))
    return np.square(pred - y).sum() / len(x)

def d_MSE(pred, y):
    # return (pred - y) / (len(pred) * 2)
    return (2/len(x)) * (pred - y)





# One Half Mean Squared Error
def halfMSE(pred, y):
    return np.square(pred - y) / (len(pred) * 2)

def d_halfMSE(pred, y):
    return (pred - y) / len(pred)



# ------ References -------
# 
# Post medium: https://medium.com/swlh/only-numpy-why-we-need-activation-function-non-linearity-in-deep-neural-network-with-529e928820bc
# Original code: https://trinket.io/python3/55549b9fea
#
