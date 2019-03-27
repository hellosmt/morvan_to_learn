import torch
import numpy as np 

a = np.arange(6).reshape((2, 3))
b = torch._C.from_numpy(a)

print(a)
print(b)
