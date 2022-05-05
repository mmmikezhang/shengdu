import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./data" ,False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)


class Zj(nn.Module):
    def __init__(self):
        super(Zj, self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output=self.maxpool1(input)
        return output


zj=Zj()
writer=SummaryWriter("logs_maxpool")
step=0
for data in dataloader:
    images,targets=data
    writer.add_images("in",images,step)
    ouput=zj(images)
    writer.add_images("out",images,step)
    step=step+1
writer.close()


        