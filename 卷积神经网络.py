"""
卷积神经网络
"""

# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F  # 加载nn中的功能函数
import torch.optim as optim  # 加载优化器有关包
import torch.utils.data as Data
# from torchvision import datasets, transforms  # 加载计算机视觉有关包
# import torchvision
# from torch.autograd import Variable
import os
from glob import glob
from PIL import Image
import numpy as np

class my_dataset(torch.utils.data.Dataset):
    def __init__(self,train_data_path,my_transform = None):
        self.img_path_list = glob(train_data_path+"/*.png")
        self.my_transform = my_transform
    def __len__(self):
        return len(self.img_path_list)
    def __getitem__(self, item):
        # print(self.img_path_list[item])
        img = Image.open(self.img_path_list[item]).convert('L')
        # l = os.path.split(self.img_path_list[item])
        label = float(os.path.split(self.img_path_list[item])[1][0])
        if self.my_transform != None:
            img = self.my_transform(img)
        else:
            img = np.asarray(img)/255.0
        return img,label
BATCH_SIZE = 64
root_path = os.getcwd()
train_img_path = r"D:\创新班\模式识别\python代码\数据库\Minist手写数字数据集\Minist手写数字数据集\Minist_img_trian\train_all"
test_img_path = r"D:\创新班\模式识别\python代码\数据库\Minist手写数字数据集\Minist手写数字数据集\Minist_img_test\test_all"
save_model_path = os.path.join(root_path,"checkpoint")
train_dataset = my_dataset(train_img_path)
test_dataset = my_dataset(test_img_path)
# 加载小批次数据，即将MNIST数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(784, 10)
    
    def forward(self, X):
        return F.relu(self.linear1(X))


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # self.linear1 = nn.Linear(784, 10)
        self.conv1 = nn.Conv2d(1,6,kernel_size=3,stride=1,padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6,16,kernel_size=3,stride=1,padding = 1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*7*7,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, X):
        x = self.conv1(X)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1,16*7*7)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
model = Model1()
loss = nn.CrossEntropyLoss()  # 损失函数选择，交叉熵函数
optimizer = optim.SGD(model.parameters(), lr=0.1)
num_epochs = 5
best_eval_acc = 0.0

torch.manual_seed(10)
torch.cuda.manual_seed(10)
for echo in range(num_epochs):
    train_loss = 0  # 定义训练损失
    train_acc = 0  # 定义训练准确度
    num_correct = 0
    model.train()  # 将网络转化为训练模式
    print(len(train_loader.dataset.img_path_list))
    for i, (X, label) in enumerate(train_loader):  # 使用枚举函数遍历train_loader
        X = torch.unsqueeze(X,dim=1)
        # X = X.view(-1, 784)  # X:[64,1,28,28] -> [64,784]将X向量展平
        X = X.float()
        label = label.long()
        # label = Variable(label)
        out = model(X)  # 正向传播
        lossvalue = loss(out, label)  # 求损失值
        optimizer.zero_grad()  # 优化器梯度归零
        lossvalue.backward()  # 反向转播，刷新梯度值
        optimizer.step()  # 优化器运行一步，注意optimizer搜集的是model的参数
        # 计算损失
        train_loss += float(lossvalue)
        # 计算精确度
        _, pred = out.max(1)
        num_correct += (pred == label).sum()
    print("epoch:" + ' ' + str(echo))
    print("train lose:" + ' ' + str(train_loss / len(train_loader)))
    print("train accuracy:" + ' ' + str(num_correct.numpy()/ len(train_loader.dataset.img_path_list)))
    eval_loss = 0
    eval_acc = 0
    num_correct = 0.0
    model.eval()  # 模型转化为评估模式
    for X, label in test_loader:
        X = torch.unsqueeze(X, dim=1)
        # X = X.view(-1, 784)
        X = X.float()
        label = label.long()
        testout = model(X)
        testloss = loss(testout, label)
        eval_loss += float(testloss)
        _, pred = testout.max(1)
        num_correct += (pred == label).sum()
    now_eval_acc = num_correct.numpy() / len(test_loader.dataset.img_path_list)
    print("test lose: " + str(eval_loss / len(test_loader)))
    print("test accuracy:" + str(now_eval_acc) + '\n')
    if now_eval_acc > best_eval_acc:
        best_eval_acc = now_eval_acc
        if os.path.exists(save_model_path) == False:
            os.makedirs(save_model_path)
        torch.save(model.state_dict(),os.path.join(save_model_path,"best_model.pt"))