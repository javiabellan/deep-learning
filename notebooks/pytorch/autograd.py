import torch
from torch.autograd import Variable


x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

y = x + 2
print(y)

z = y * y * 3
print(z)

out = z.mean()
print(out)

#######################

out.backward() # let’s backprop 


print(x.grad) # print gradients d(out)/dx