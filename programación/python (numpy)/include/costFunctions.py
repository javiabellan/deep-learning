#########################################################
#
#						FUNCIONES DE ERROR
#
#	Sin regularización
#
#		L1 norm loss
#		L2 norm loss 
#	
#	Con regularización
#
#		L1 norm loss + L1 regularization 
#		L2 norm loss + L2 regularization
#		L1 norm loss + L2 regularization
#		L2 norm loss + L1 regularization
#	
#	https://becominghuman.ai/only-numpy-implementing-different-combination-of-l1-norm-l2-norm-l1-regularization-and-14b01a9773b
#
#########################################################






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