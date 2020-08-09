"""
多层感知机
"""
import cv2 as cv
from glob import glob
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch import optim
from tqdm import tqdm


train_img_path = r"D:\创新班\模式识别\python代码\数据库\Minist手写数字数据集\Minist手写数字数据集\Minist_img_trian\train_all"
test_img_path = r"D:\创新班\模式识别\python代码\数据库\Minist手写数字数据集\Minist手写数字数据集\Minist_img_test\test_all"


class my_datasets(Dataset):
    """
    类描述：定义的数据类
    """
    def __init__(self,img_path):
        # super(my_datasets).__init__()
        self.img_path = img_path
        self.img_path_lists = glob(self.img_path+"\\*.png")
    def __len__(self):
        return len(self.img_path_lists)
    def __getitem__(self, item):
        img_path = self.img_path_lists[item] ###获取图片路径
        ##打开图片并转换为灰度图
        img = Image.open(img_path).convert('L')
        #转换为numy类型的数据格式
        img_numpy = np.asarray(img)
        img_numpy = img_numpy/255.0  ##进行归一化
        label = float(img_path.split("\\")[-1][0])  ##获取图像标签
        return img_numpy,label

class my_Model(torch.nn.Module):  ###__call__
    def __init__(self):
        super(my_Model, self).__init__()
        self.one_layer = torch.nn.Linear(28*28,10)
        # pass
    def forward(self,input):
        output = self.one_layer(input)
        return output    ##leetcode

if __name__ == "__main__":
    train_datasets = my_datasets(train_img_path)  ##训练集数据
    test_datasets = my_datasets(test_img_path)   ###测试集数据
    ###6400->3  6400/batch_szie = 100
    epochs = 3
    batch_size = 64
    base_lr =0.1
    ##训练集数据容器
    train_loader = DataLoader(dataset = train_datasets,batch_size = batch_size,shuffle = True)
    ##测试集数据容器
    test_loader = DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=True)
    net = my_Model()
    citer_loss = torch.nn.CrossEntropyLoss()  ###交叉熵损失函数 -ylog(p)-(1-y)log(1-p)
    pp_optim = optim.SGD(net.parameters(),lr = base_lr)     ##定义优化器
    now_acc = 0.0
    best_acc = 0.0
    for epoch in range(epochs):
        print("**"*20)
        print("epoch===》",epoch)
        print("**"* 20)
        net.train()
        # i  = 0
        print("train")
        for train_img,label in train_loader:
            # if i == 1:
            #     break
            # i +=1
            # train_img=>(64*28*28)
            train_img = torch.Tensor.view(train_img,(-1,28*28)).float()
            label = label.long()
            out = net(train_img)
            loss = citer_loss(out,label)
            
            pp_optim.zero_grad()
            loss.backward()
            pp_optim.step()
            # print(loss)
        net.eval()
        correct = 0.0
        print("eval")
        with torch.no_grad():
            print("样本个数",len(test_loader.dataset.img_path_lists))
            for test_img,test_label in test_loader:
                test_img = torch.Tensor.view(test_img,(-1,28*28)).float()
                test_label = test_label.long()
                
                out = net(test_img)
                _,pres = torch.max(out,dim=1)   ##pres->预测标签
                correct = correct + (pres==test_label).sum()  ###计算所有正确样本
            acc  =  correct.numpy()/len(test_loader.dataset.img_path_lists)    ##计算准确率
            print("acc = ",acc)
            print("finish eval data")
                
        
        # out = my_Model()
        # pass
    # train_datasets[0]
    # pass