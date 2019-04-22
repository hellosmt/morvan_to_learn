import torch


# 所有输入的requires_grad都为False，输出才为False
a = torch.randn(5, 5)
print(a.requires_grad)  # False

b = torch.randn(5, 5)
y = a+b
print(y.requires_grad)  # False

c = torch.randn((5, 5), requires_grad=True)
y = c + a
print(y.requires_grad)  # True


