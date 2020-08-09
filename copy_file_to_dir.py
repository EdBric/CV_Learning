import shutil
import os
from glob import glob
from tqdm import tqdm       #Tqdm 是一个快速，可扩展的Python进度条
trian_img_path = r"D:\创新班\模式识别\python代码\数据库\Minist手写数字数据集\Minist手写数字数据集\Minist_img_trian\Minist_img_trian"
train_all_save_path = r"D:\创新班\模式识别\python代码\数据库\Minist手写数字数据集\Minist手写数字数据集\Minist_img_trian\train_all"
test_img_path = r"D:\创新班\模式识别\python代码\数据库\Minist手写数字数据集\Minist手写数字数据集\Minist_img_test\Minist_img_test"
test_all_save_path = r"D:\创新班\模式识别\python代码\数据库\Minist手写数字数据集\Minist手写数字数据集\Minist_img_test\test_all"
"""
copy_data（）
函数描述：复制文件到指定路径
trian_img_path：图片路径
train_all_save_path：图片保存路径
"""
def copy_data(trian_img_path,train_all_save_path):
    if os.path.exists(train_all_save_path) == False:  #os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径。
        os.makedirs(train_all_save_path)              #os.makedirs() 方法用于递归创建目录。
    train_dir_list = os.listdir(trian_img_path)       #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。【返回文件名】
    for i in tqdm(range(len(train_dir_list))):        #遍历文件夹列表。tqdm（）显示进度条。
        """
        os.path.join()函数：连接两个或更多的路径名组件
        1.如果各组件名首字母不包含’/’，则函数会自动加上
        2.如果有一个组件是一个绝对路径，则在它之前的所有组件均会被舍弃
        3.如果最后一个组件为空，则生成的路径以一个’/’分隔符结尾
        """
        pp = os.path.join(trian_img_path,train_dir_list[i])
        img_list = glob(os.path.join(pp,"*.png"))     #os.path.join()将目录和文件名合成一个路径;glob()得到指定路径下指定格式的文件【返回全路径】
        num_t = 0
        for j in range(len(img_list)):
            shutil.copy(img_list[j],os.path.join(train_all_save_path,train_dir_list[i]+"_"+str(num_t)+".png"))
            """
            shutil.copy(src, dst)
            复制文件 src 到 dst 文件或文件夹中。 
            如果 dst 是文件夹， 则会在文件夹中创建或覆盖一个文件，且该文件与 src 的文件名相同。 
            文件权限位会被复制。
            使用字符串指定src 和 dst 路径。
            """
            num_t +=1
    pass
if __name__ == "__main__":
    print("**"*20)
    print("copy train")
    print("**" * 20)
    # copy_data(trian_img_path,train_all_save_path)
    print("**"*20)
    print("copy test")
    print("**" * 20)
    copy_data(test_img_path, test_all_save_path)

