import torch
import torch.nn.functional as F




# transform = transforms.Compose([
#     transforms.ToTensor(),                  # 将图片转变为C*W*H
#     transforms.Normalize((0.1307,),(0.3081,))       #对0-255色值进行归一化处理，其中0.1307是均值，0.3081是标准差
#
# ])

kerner = 3

#定义模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 10, kernel_size=kerner)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=kerner)
        self.conv3 = torch.nn.Conv2d(20, 10, kernel_size=kerner)
        self.pooling = torch.nn.MaxPool2d(2)
        self.linear1 = torch.nn.Linear(12250, 2000)  # 运行时这里需要手动更改
        self.linear2 = torch.nn.Linear(2000, 1000)
        self.linear3 = torch.nn.Linear(1000, 300)
        self.linear4 = torch.nn.Linear(300, 8)




    def forward(self,x):
        bach_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        # print("x.size:", x.shape)
        x = x.view(bach_size, -1)
        # print("x.size_afterView:",x.shape)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        y_hat = self.linear4(x)
        return y_hat



