"""
处理joson文件
author: LJQ
"""
import json
import os

import torch
from matplotlib.pyplot import imread
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

import numpy as np


class Mydataset(Dataset):


    def __init__(self, root, mode, csv_path,transformer=True): # 初始化各个变量
        super(Mydataset, self).__init__()
        self.root = root
        self.mode = mode
        self.csv_path = csv_path
        self.transformer = transformer
        self.raw_lables, self.raw_image_id = self.get_train_data()  # 得到类别和图像id,用于创建按csv文件
        self.datalen = len(self.raw_lables)     # 获取整个数据集的长度
        self.image_id, self.lable = self.get_csvdata()  # 得到图像id和动作类别，用于训练和测试
        # 对数据集进行划分
        if mode == "train":  # 80%
            self.image_id = self.image_id[:int(0.8 * self.datalen)]
            self.lable = self.lable[:int(0.8 * self.datalen)]
        elif mode == "test": # 20%
            self.image_id = self.image_id[int(0.8 * self.datalen):]
            self.lable = self.lable[int(0.8 * self.datalen):]
        elif mode == "val": # 100%
            self.image_id = self.image_id
            self.lable = self.lable
        else:
            assert False,"mode输入错误"



    def __getitem__(self, idx): # 装载数据，返回[img,label]
        img_path = os.path.join(self.root, self.image_id[idx]) # 获取图片路径
        # print(img_path)
        assert os.path.exists(img_path),"getitem中图像路径不存在"
        img = imread(img_path)  # 读取图片
        img = torch.tensor(img)
        if self.transformer == True:
            tf = transforms.Compose([
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),  # 对0-255色值进行归一化处理，其中0.1307是均值，0.3081是标准差
                transforms.Resize([224,224])        # 更改图片大小为224
            ])
            img = transforms.Resize([300,3])(img)  # 更改通道数
            img = torch.transpose(img, 0, 2) # 交换维度,变为C×W×H
            img = tf(img)


        lable = torch.tensor(int(self.lable[idx]))  # 将对应的lable转变为tensor
        # print(lable)
        return img,lable

    def __len__(self):
        return len(self.image_id)


    def get_csvdata(self):
        """
        读取csv文件
        :return: image_id label_id
        """
        image_id = []
        label = []
        assert os.path.exists(self.csv_path),"data.csv文件不存在"
        file = open(self.csv_path,'r')
        file = file.readlines()
        for i in range(self.datalen):        # 进行划分数据
            image_id.append(file[i].split(' ')[0])
            label.append(file[i].split(' ')[1][0])

        return image_id, label


    def get_train_data(self):
        """
        处理思路：
            1. 通过读取遍历json文件，得到图像名称和路径。
        :param josn_path: 数据集的json文件
        :return: labels,images_id 哦返回标签和图像id
        """
        json_path = "./data/dataset/detail.json"
        img_path = "/data/thumbnail_images"

        assert os.path.exists(json_path), "路径不存在"

        labels = []
        images_id = []  # 图像数组和标签

        fp = open(json_path, 'r', encoding='utf8')  # 打开json文件
        raw_data = json.load(fp)  # 读取json文件为字符串
        for i in range(len(raw_data)):
            labels.append(raw_data[i]["industy_category"])
            images_id.append(raw_data[i]["thumbnail"])
        # print(len(set(labels))) # 通过集合判断类别的个数
        # 通过集合判断类别的个数print(images_id)
        return labels, images_id



    def get_lable_id(self):
            """
            生成csz文件
            :return:lable_id
            """
            lable_id = []
            # lable_list = list(set(self.lables)) # 将lable列表转换为集合,再转换为list,以便于排序
            lable_list = ['产业概述','产业链','商业模式','市场容量','技术变革', '政策法规', '竞争格局', '行业数据' ] # 将lable列表转换为集合,再转换为list,以便于排序
            # print(lable_list)
            # print(self.lables)
            for i in range(len(self.raw_lables)):    # 便利原始数据集
                if self.raw_lables[i] in lable_list:    # 返回集合对应的下标
                    lable_id.append(lable_list.index(self.raw_lables[i]))
            # 写入csz 文件
            for i in range(len(lable_id)):
                f = open('./data/dataset/data.csv','a')
                f.write(str(self.raw_image_id[i]))
                f.write(' ')
                f.write(str(lable_id[i]))
                f.write('\n')
            f.close()
            return True

if __name__ == '__main__':
    csvpath = './data/dataset/data.csv'
    db = Mydataset(os.path.abspath('./data/dataset/thumbnail_images'),"train", csvpath)
    print(db.__getitem__(2))