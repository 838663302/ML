import torch
import numpy as np

x = torch.randint(0,10,(2,2,2))
print(x.long())
print(torch.tensor(1).shape)
print(x.device)
y = torch.Tensor([2,3])
print(y.dtype)
print(y)
print(torch.tensor([2,3]).dtype)
print(torch.linspace(0,5,7))
z = torch.full((2,3), 6)
print(z)
print(torch.full_like(z, 7))
print(torch.empty_like(z))
torch.LongTensor([1,2,3])
print(torch.tensor(np.random.rand(2,3)))
print(torch.tensor(np.random.randint(0,10,(2,3))).float().exp())
