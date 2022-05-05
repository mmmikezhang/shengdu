import torchvision
from torch.utils.tensorboard import SummaryWriter
import time


# 搭建神经网络
import torch
from torch import nn


class Zj(nn.Module):
    def __init__(self):
        super(Zj, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            # nn.ReLU(),
            nn.Linear(64,10),
            # nn.Softmax()

        )
    def forward(self,x):
        x=self.model(x)
        return x

# 准备数据
from torch import nn
from torch.utils.data import DataLoader





train_data=torchvision.datasets.CIFAR10(root="./data",train=True,transform=torchvision.transforms.ToTensor())
test_data=torchvision.datasets.CIFAR10(root="./data",train=False,transform=torchvision.transforms.ToTensor())

train_data_size=len(train_data)
test_data_size=len(test_data)
# 格式化字符串：
# 训练数据长:10
print("训练数据长:{}".format(train_data_size))
print("测试数据长:{}".format(test_data_size))

# 利用dataloader来加载数据
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)


# 创建网络模型
zj=Zj()

if torch.cuda.is_available():

    zj=zj.cuda()

# 定义损失函数
loss_fn=nn.CrossEntropyLoss()

if torch.cuda.is_available():
    loss_fn=loss_fn.cuda()

# 优化器
#  1e-2=1/(10*10)=0.01
leaning_rate=1e-2
optimizer=torch.optim.SGD(zj.parameters(),leaning_rate)




# 设置训练的一些参数
# 记录训练次数 total_train_step
total_train_step=0
# 记录测试的次数 total_test_step
total_test_step=0
#训练轮数 epoch
epoch=10

start_time=time.time()  #开始时间

#  添加tensorboard进行数据可视化
writer=SummaryWriter("./logs_train")


for i in range(epoch):
    print("-----第{}轮训练开始----".format(i+1))


#     训练步骤开始
    for data in train_dataloader:

        imgs,targets=data

        if torch.cuda.is_available():
            imgs=imgs.cuda()
            targets=targets.cuda()

        outputs=zj(imgs)

        loss=loss_fn(outputs,targets)

        optimizer.zero_grad()  #梯度清零
        loss.backward()  #反向传播得到参数间的梯度
        optimizer.step() #对参数进行优化

        total_train_step=total_train_step+1
        if total_train_step%100==0:

            end_time=time.time()

            print("训练次数：{}，训练时间：{}".format(total_train_step,end_time-start_time))   #记录时间用

            print("训练次数：{}，loss：{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
# 需要参数global_step，填训练次数
#测试步骤开始
    total_test_loss=0

    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data

            if torch.cuda.is_available():
                imgs=imgs.cuda()
                targets=targets.cuda()
            outputs=zj(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy=total_accuracy+accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))    # 记录测试的次数

    print("整体测试集上的accuracy：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)

    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)

    total_test_step=total_test_step+1

# 对每一轮的模型进行保存
    torch.save(zj,"zj_{}.pth".format(i))
    print("模型已保存")

writer.close()




