import torch

x = torch.ones(2, requires_grad=True)  # 表示跟踪它的计算 后面要用到它的梯度,默认是False，如果指定为True，则后面依赖它的所有节点都为True
print("\nx:", x)
print("\nx.grad_fn:", x.grad_fn)  # 用户自己定义的，没有这个属性
y = x + 2   # y是由计算操作得到的，不是用户自己定义的，因此有grad_fn的属性
print("\ny:", y)
print("y.grad_fn:", y.grad_fn)
z = y*y
out = z.mean()
out.backward()  # 反向传播，相当于执行out.backward(torch.tensor(1.))
print(x.grad)

# 使用torch.no_grade()
print(x.requires_grad)
print((x**2).requires_grad)
with torch.no_grad():
    print((x**2).requires_grad)
    print(x.requires_grad)  # 这里为什么是True

# 教程里说的y不再是标量没有看明白
