import torch


# PyTorch中所有的神经网络都来自于autograd包
x = torch.ones((2, 2), requires_grad=True)  # 表示跟踪它的计算 后面要用到它的梯度,默认是False，如果指定为True，则后面依赖它的所有节点都为True
print("\nx:", x)
print("\nx.grad_fn:", x.grad_fn)  # 用户自己定义的，没有这个属性
y = x + 2   # y是由计算操作得到的，不是用户自己定义的，因此有grad_fn的属性
print("\ny:", y)
print("y.grad_fn:", y.grad_fn)
z = y*y*3
out = z.mean()
out.backward()  # 反向传播，相当于执行out.backward(torch.tensor(1.))
print("\nx.grad：", x.grad)

# 使用torch.no_grade() 停止跟踪计算
print(x.requires_grad)
print((x**2).requires_grad)
with torch.no_grad():
    print((x**2).requires_grad)
    print(x.requires_grad)  # 这里为什么是True

# 教程里说的y不再是标量没有看明白

a = torch.randn(3, requires_grad=True)
print("\na:", a)
b = a*2
print("\nb:", b)
while b.data.norm()<1000:
    b = b*2
print("\nb:", b)

w = torch.tensor([0.1, 0.001, 0.001], dtype=torch.float) # 与y的维度是一样的 变成标量再求导 不允许tensor对tensor进行求导
b.backward(w)
print("\na.grad:", a.grad)

