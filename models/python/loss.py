'''
Loss functions

Sin regularización

	L1 norm loss
	L2 norm loss
	Cross-Entropy
	Hinge
	Huber
	Kullback-Leibler
	L1
	L2
	Maximum Likelihood
	Mean Squared Error
	
Con regularización

	L1 norm loss + L1 regularization 
	L2 norm loss + L2 regularization
	L1 norm loss + L2 regularization
	L2 norm loss + L1 regularization
	
	https://becominghuman.ai/only-numpy-implementing-different-combination-of-l1-norm-l2-norm-l1-regularization-and-14b01a9773b
	http://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
'''





################################################### Absolute error (L1)
def L1(pred, y):
    return np.sum(np.absolute(pred - y))


################################################### Squared error (L2)
def L2(pred, y):
    return np.sum((pred - y)**2)


################################################### Mean Squared Error (MSE)
def MSE(pred, y):
	# return ((pred - y) ** 2).mean(axis=ax)
	# return np.square(pred - y).sum() / len(y)
    return np.sum((pred - y)**2) / y.size

def d_MSE(pred, y):
	# return pred - y
    # return (pred - y) / (len(pred) * 2)
    return (2/len(x)) * (pred - y)


################################################### One Half Mean Squared Error
def halfMSE(pred, y):
    return np.square(pred - y) / (len(pred) * 2)

def d_halfMSE(pred, y):
    return (pred - y) / len(pred)


################################################### Cross entropy (categorical)
def categoricalCrossEntropy(pred, y):
	if y == 1:
		return -log(pred)

################################################### Cross entropy (binary)
def binaryCrossEntropy(pred, y):
	if y == 1:
		return -log(pred)
	else:
		return -log(1 - pred)

################################################### Maximum Likelihood
def maximumLikelihood(pred, y):
    pass

################################################### Hinge
def Hinge(pred, y):
    return np.max(0, 1 - pred * y)

################################################### Huber
def Huber(pred, y):
    pass

################################################### kullback Leibler divergence
def KLDivergence(pred, y):
    pass

