import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=1)


class Zj(nn.Module):
    def __init__(self):
        super(Zj, self).__init__()
        self.model1 = Sequential(
             Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
             MaxPool2d(kernel_size=2),
             Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
             MaxPool2d(kernel_size=2),
             Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
             MaxPool2d(kernel_size=2),
             Flatten(),
             Linear(in_features=1024, out_features=64),
             Linear(in_features=64, out_features=10)

        )
    def forward(self,x):
       x=self.model1(x)
       return x

loss=nn.CrossEntropyLoss()
zj=Zj()
for data in dataloader:
    imgs,targets=data
    # torch.reshape(imgs,(-1,3,32,8))l
    outputs=zj(imgs)
    # print(outputs)
    # print(targets)
    result_loss=loss(outputs,targets)
    print(result_loss)
