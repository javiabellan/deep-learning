import torch
import torch.nn as nn
import torch.nn.functional as F


########################################  TENSORES

x = torch.tensor([5.5, 3]) 
x = torch.empty(5, 3)      # Garbage
x = torch.rand(5, 3)       # Random
x = torch.zeros(5, 3)      # Zeros
x = torch.ones(2, 2)       # Ones


x.data
x.device
x.dtype
x.shape = x.size()

x = torch.ones(2, 2, requires_grad=True) # With gradients
out = x.pow(2).sum()
out.backward()
x.grad