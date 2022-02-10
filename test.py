import torch


# 测试函数
from train import model


def test(test_loader):
    with torch.no_grad():
        correct = 0;
        total = 0
        with torch.no_grad():
            for i,data in enumerate(test_loader,0):
                inputs,lables = data
                # 将输入迁移到gpu上
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                inputs = inputs.to(device)
                lables = lables.to(device)
                y_hat = model(inputs)
                maxNum, pre_site = torch.max(y_hat ,dim = 1)
                total += lables.size(0)
                correct += (pre_site == lables).sum().item()
        print("模型准确度为：",100 * correct / total,"%")