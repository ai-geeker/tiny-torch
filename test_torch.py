import torch
import torch.nn.functional as F
a = torch.ones(4, requires_grad=True)
b = torch.ones(4, requires_grad=True)

y = torch.sin(2 * a + b) + torch.cos(a)
print("y", y)
y = F.relu(y)
y = y.sum()
print("y", y)
y.backward()

print("a.grad", a.grad)
print("b.grad", b.grad)