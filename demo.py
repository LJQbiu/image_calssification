import torch
import os

from matplotlib.image import imread
from torchvision import transforms


def demo(img_path):
    """
    用于demo，输入图片输出类别
    :return:
    """
    assert os.path.exists(img_path),"输入图片不存在"

    # demo处理数据，先将就着用，未来再改进
    img = imread(img_path)
    img = torch.tensor(img)
    tf = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # 对0-255色值进行归一化处理，其中0.1307是均值，0.3081是标准差
    img = tf(img)
    img = torch.tensor(img) # 将图片转换为张量

    pkl_path = './work_dir/model0.pkl'
    assert os.path.exists(pkl_path),"权重文件不存在"
    model = torch.load(pkl_path,map_location='cpu')
    print("模型加载成功")
    img = torch.unsqueeze(img,0)      # 扩展维度

    y_hat = model(img).argmax().item()
    predic_class = get_label_class(y_hat)
    return y_hat,predic_class

def get_label_class(lable_id):
    f = open('data/class.txt','r',encoding="utf8")  # 读取类别文件
    lable_list = f.readlines()
    # print(lable_list)
    lable_text= lable_list[lable_id]
    return lable_text


if __name__ == '__main__':
    img_name = '503_thumbnail.png'
    img_path = os.path.join("./data/val",img_name)
    y_hat, pre_class = demo(img_path)
    print(y_hat,pre_class)