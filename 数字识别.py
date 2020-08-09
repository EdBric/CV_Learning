import torch
from PIL import Image
import numpy as np
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(784, 10)
    
    def forward(self, X):
        return F.relu(self.linear1(X))
##

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # self.linear1 = nn.Linear(784, 10)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, X):
        x = self.conv1(X)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
if __name__ == "__main__":
    net = Model1()
    test_img_path = r"D:\创新班\模式识别\python代码\数据库\Minist手写数字数据集\Minist手写数字数据集\Minist_img_test\test_all\11.png"
    net.load_state_dict(torch.load(r"D:\创新班\模式识别\python代码\数字识别\checkpoint\best_model.pt"))
    net.eval()
    with torch.no_grad():
        img = Image.open(test_img_path).convert("L")
        img_numpy = np.asarray(img)/255.0
        img_numpy = np.expand_dims(np.expand_dims(img_numpy,axis=0),axis=0)
        # img_numpy = np.reshape(img_numpy,(1,784))
        img_tensor = torch.Tensor(img_numpy)
        pre = net(img_tensor)
        _,ll = torch.max(pre,dim=1)
        print("此图片所属类别为：",ll.numpy()[0])