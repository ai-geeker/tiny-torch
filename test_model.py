
import tinytorch as torch
import numpy as np
import tinytorch.nn.functional as F


model = torch.nn.Linear(10, 4)
a = torch.ones(10, 30)


x = model(a)
y = F.relu(a)
y = x.sum()
print("y", y)