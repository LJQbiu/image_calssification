import torch
import torch.optim as optim
from model.resnet import resnet18

# model = Net()
model = resnet18(3,8)
# 指定使用设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

critirion = torch.nn.CrossEntropyLoss(size_average=True)
optimizim = optim.SGD(model.parameters(), lr = 0.1, momentum=0.5)


# 改进后的train函数
def train(epoch,train_loader):
    loss_sum = 0.0
    for i,data in enumerate(train_loader,0):
        inputs, lables = data
        inputs = inputs.to(device)
        lables = lables.to(device)
        y_hat = model(inputs)
        loss = critirion(y_hat,lables)
        optimizim.zero_grad()
        loss.backward()
        loss_sum += loss.item()
        optimizim.step()
        if i%100 == 99:
            print(epoch+1,i+1,loss_sum/100)
            loss_sum = 0.0
    #  模型保存
    torch.save(model, ('./work_dir/model'+str(epoch+1))+'.pkl')