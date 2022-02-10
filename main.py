"""
程序入口
"""
import os
from torch.utils.data import DataLoader
from data import Mydataset
from test import test
from train import train

batch_size = 1
# 加载数据
img_root = os.path.abspath('./data/dataset/thumbnail_images')
csv_path = os.path.abspath('./data/dataset/data.csv')
train_dataset = Mydataset(img_root,"train",csv_path)
test_dataset = Mydataset(img_root,"test",csv_path)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=2)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=2)

if __name__ == '__main__':
    for epoch in range(9):
        train(epoch,train_loader)
        test(test_loader)
