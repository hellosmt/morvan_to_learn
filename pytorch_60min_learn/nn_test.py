import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):  # 将Net作为nn.Module的子类
    # 定义Net的初始化函数
    def __init__(self):
        super(Net, self).__init__()  # 调用父类的初始化函数
        self.conv1 = nn.Conv2d(1, 6, 5)  # conv1函数为图像卷积函数，输入为1个channel,也就是灰度图，输出为6个channel，卷积核大小5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)  # 原图像是32x32的，经过前面的卷积以及待会的pooling之后为16x5x5，所以这里的输入节点个数为400
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 10)

    # 定义前向传播的过程，这个是必须的，一旦定义成功，反向传播的过程是自动求导实现的
    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), (2, 2))  # 使用2x2的窗口大小进行最大池化操作，更新x
        x = F.max_pool2d(self.conv2(x), 2)  # 如果窗口大小是一个正方形的，就可以只指定一个数字
        x = x.view(-1, self.flatten(x))  #拉直，通过flatten函数可以得到特征的数目，-1表示不管有多少行，出来的就是一个batch有几个图片
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def flatten(self, x):
        size = x.size()[1:]  # 除开batch_size那一维，剩下的维度是我们要算的特征的个数
        num_features = 1
        for i in size:
            num_features = num_features * i
        return num_features

net = Net()   # 实例化一个Net
print(net)  # 查看网络结构

params = list(net.parameters())
print(len(params))  # 这里是10，因为有两个卷积层三个全连接层，每层都有W和bias，所以一共十个
print(params[0].size())  # 就是第一个卷积层的参数

# 自己定义的输入
input = torch.randn(1, 1, 32, 32)
output = net.forward(input)  # 进行前向传播得到输出
print("\noutput:", output)

# 在反向传播之前需要将所有参数的梯度缓存清零 否则是不断累加的
#net.zero_grad()
#output.backward(torch.randn(1, 10))  # 这里不明白 为什么要传一个张量进去
# torch.nn只支持小批量的输入，如果只输入单个样本，需要使用input.unsqueeze(0)，来添加它的维数

# 接下来开始损失函数和更新权值
# 损失函数接收一对output和target，target就是真实标签值
target = torch.randn(1, 10)  # 自己设置的真实标签值
criterion = nn.MSELoss()  # 选择一个损失函数 这里是均方误差
loss = criterion(output, target)
print("\nloss:", loss)
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])  # 反向跟踪loss 它的loss前一步是linear

# 开始反向传播
net.zero_grad()  # 清除所有参数的grad butffer
print("\nconv1.bias.grad before bp: ", net.conv1.bias.grad)

loss.backward()  # 计算dloss/dinput

print("\nconv1.bias.grad after bp: ", net.conv1.bias.grad)
