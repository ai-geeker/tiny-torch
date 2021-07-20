from tinytorch import Tensor
import tinytorch as torch
import numpy as np
import tinytorch.nn.functional as F

a = torch.ones(4, requires_grad=True)
b = torch.ones(4, requires_grad=True)

n = np.ones(4)
n = n + 1

y = torch.sin(2 * a + b) - torch.cos(a)
y = F.relu(y)
print("y", y)
y = y.sum()
print("y", y)
y.backward()
print("a", a)
print("b", b)

