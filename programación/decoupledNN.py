#################################################
#
#   Decoupled nn
#
#   Paper: https://arxiv.org/abs/1608.05343
#   Code: https://gitlab.com/jae.duk.seo/Only_Numpy/tree/master/2_Decoupled_Neural_Network
#
#################################################

import numpy as np
import sys,time




def generate_dataset(output_dim = 8,num_examples=1000):
    def int2vec(x,dim=output_dim):
        out = np.zeros(dim)
        binrep = np.array(list(np.binary_repr(x))).astype('int')
        out[-len(binrep):] = binrep
        return out

    x_left_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    x_right_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    print(x_left_int[0])
    print(x_right_int[0])
    y_int = x_left_int + x_right_int
    print(y_int[0])

    x = list()
    for i in range(len(x_left_int)):
        x.append(np.concatenate((int2vec(x_left_int[i]),int2vec(x_right_int[i]))))

    y = list()
    for i in range(len(y_int)):
        y.append(int2vec(y_int[i]))

    x = np.array(x)
    y = np.array(y)
    
    return (x,y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# the special way of derivative
def sigmoid_out2deriv(out):
    return out * (1 - out)


np.random.seed(1234)

num_examples = 1000
output_dim = 12
iterations = 2000

x,y = generate_dataset(num_examples=num_examples, output_dim = output_dim)

batch_size = 1000
alpha = 0.03


class DNI(object):
    
    def __init__(self,input_dim, output_dim,nonlin,nonlin_deriv,alpha ):
        
        self.weights = (np.random.randn(input_dim, output_dim) * 0.2) - 0.1
        self.weights_synthetic_grads = (np.random.randn(output_dim,output_dim) * 0.2) - 0.1
        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv
        self.alpha = alpha
    
    def forward_and_synthetic_update(self,input):
        
        # Forward
        self.input = input
        self.output = self.nonlin(self.input.dot(self.weights))
        
        # Backward
        self.synthetic_gradient = self.output.dot(self.weights_synthetic_grads)
        self.weight_synthetic_gradient = self.synthetic_gradient * self.nonlin_deriv(self.output)
        self.weights += self.input.T.dot(self.weight_synthetic_gradient) * self.alpha
        
        return self.weight_synthetic_gradient.dot(self.weights.T), self.output
    
    def update_synthetic_weights(self,true_gradient):
        self.synthetic_gradient_delta = self.synthetic_gradient - true_gradient
        self.weights_synthetic_grads += self.output.T.dot(self.synthetic_gradient_delta) * self.alpha





#  input = 24, output = 12 , layer_1_dim = 128, layer_2_dim = 64
start = time.time()
input_dim = len(x[0])
layer_1_dim = 128
layer_2_dim = 64
output_dim = len(y[0])
layer_1 = DNI(input_dim,layer_1_dim,sigmoid,sigmoid_out2deriv,alpha)
layer_2 = DNI(layer_1_dim,layer_2_dim,sigmoid,sigmoid_out2deriv,alpha)
layer_3 = DNI(layer_2_dim, output_dim,sigmoid, sigmoid_out2deriv,alpha)


for iter in range(iterations):
    error = 0

    batch_x = x
    batch_y = y
    
    _, layer_1_out = layer_1.forward_and_synthetic_update(batch_x)
    layer_1_delta, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out)
    layer_1.update_synthetic_weights(layer_1_delta)
    
    layer_2_delta, layer_3_out = layer_3.forward_and_synthetic_update(layer_2_out)
    layer_2.update_synthetic_weights(layer_2_delta)

    layer_3_delta =  layer_3_out - batch_y
    layer_3.update_synthetic_weights(layer_3_delta)

    # This is the true update of the gradient
    # layer_3.update_synthetic_weights(layer_3_delta)
    # layer_2.update_synthetic_weights(layer_2_delta)
    # layer_1.update_synthetic_weights(layer_1_delta)
    
    error += (np.sum(np.abs(layer_3_delta * layer_3_out * (1 - layer_3_out))))

    if(error < 0.1):
        sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error))
        break       
        
    sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error))
    if(iter % 100 == 0):
        print("")

end = time.time()


_, layer_1_out = layer_1.forward_and_synthetic_update(x)
layer_1_delta, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out)
layer_2_delta, layer_3_out = layer_3.forward_and_synthetic_update(layer_2_out)

for iter in range(10):
    print(x[iter][:12].dot(2**np.arange(x[iter][:12].size)[::-1])  )
    print(x[iter][12:].dot(2**np.arange(x[iter][:12].size)[::-1])  )
    print("-----------")
    print(layer_3_out[iter].dot(2**np.arange(x[iter][:12].size)[::-1]))

    truteh = x[iter][:12] + x[iter][12:]
    print("The truth data: ",truteh.dot(2**np.arange(x[iter][:12].size)[::-1]),'\n')
    
print("\n\n------------\nTraining Time: ",end - start )
