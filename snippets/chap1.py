from __future__ import print_function
import torch

x = torch.zeros(5, 3, dtype=torch.long)
# x = torch.tensor([4, 3.2])
x = x.new_ones(5, 3, dtype=torch.double)

# print(x)

x = torch.randn_like(x, dtype=torch.float)

print(x)

result = torch.empty(5, 3)
y = torch.rand(5, 3)
y1 = torch.add(x, y, out=result)
print(y)

print(x + y)
print(result)

y.add_(x)
print(y.numpy())
