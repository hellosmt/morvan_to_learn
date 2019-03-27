import torch
import numpy as np

#从数据中直接构建一个张量：
x = torch.tensor([5.5, 3, 2, 111])
print("\nx:", x)

#从已有张量去构建一个新的张量 重用输入张量的属性 除非提供新值 如dtype
x_new = x.new_ones(5, 3) #这种new_*的方法参数为新张量的size
print("\nx_new:", x_new)
x_new = torch.randn_like(x, dtype=torch.float) #这种方法的参数是原来的张量
print("\nx_new:", x_new)

#获得张量的size，torch.Size实际上是一个元组，所以支持元组的所有操作
print("\nx_new的size:", x_new.size())

#操作语法 加法为例
y = torch.ones(1, 4)
print("\nx+y...1:", x+y)

print("\nx+y...2:", torch.add(x, y))

#给出一个输出张量作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print("\nx+y...3:", result)

#原地操作 in-place，任何原地操作改变张量的操作都有一个_后缀
y.add_(x)  #把x加到y上去，这样会改变y的值
print("\nx+y:...4:", y)

#numpy的索引功能随便用
print("\n切片：", x[0:2])

#修改张量的size
a = torch.rand(4, 5)
b = a.view(-1, 10)  #-1表示没有指定维度
print("\na:", a)
print("\nb:", b)

#单元素张量使用.item()
c = torch.rand(1)
print("\nc:",c)
print("\nc.item():", c.item())

#numpy桥
##将tensor转为numpy数组
a = torch.ones(5)
print('\na:', a)

b = a.numpy()
print('\nb:', b)

a.add_(1)
print('\na:', a)
print('\nb:', b)  #b也加了1，a和b是共享潜在内存的，a变b也会变

##numpy变成tensor
a = np.ones(5)
b = torch.from_numpy(a)
print("\na:", a)
print('\nb:', b)
np.add(a, 1, out=a)
print("\na:", a)
print('\nb:', b)

