import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):  # 将Net作为nn.Module的子类
    # 定义Net的初始化函数
    def __init__(self):
        super(Net, self).__init__()  # 调用父类的初始化函数
        self.conv1 = nn.Conv2d(3, 6, 5)  # conv1函数为图像卷积函数，输入为1个channel,也就是灰度图，输出为6个channel，卷积核大小5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 原图像是32x32的，经过前面的卷积以及待会的pooling之后为16x5x5，所以这里的输入节点个数为400
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 10)

    # 定义前向传播的过程，这个是必须的，一旦定义成功，反向传播的过程是自动求导实现的
    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), (2, 2))  # 使用2x2的窗口大小进行最大池化操作，更新x
        x = F.max_pool2d(self.conv2(x), 2)  # 如果窗口大小是一个正方形的，就可以只指定一个数字
        x = x.view(-1, self.flatten(x))  # 拉直，通过flatten函数可以得到特征的数目，-1表示不管有多少行，出来的就是一个batch有几个图片
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


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("1.jpg")


# show some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s'%classes[labels[j]] for j in range(4)))


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

# 训练
for epoch in range(2):
    loss_total = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()
        output = net.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        loss_total = loss_total+loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss_total / 2000))
            loss_total = 0.0

