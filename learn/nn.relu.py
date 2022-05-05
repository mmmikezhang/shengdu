import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor())

dataloader=DataLoader(dataset,batch_size=64)
class Zj(nn.Module):
    def __init__(self):
        super(Zj, self).__init__()
        self.relu1=ReLU()
        self.sigmoid1=Sigmoid()
    def forward(self,input):
        # output=self.relu1(input)
        output=self.sigmoid1(input)
        return output
zj=Zj()

writer=SummaryWriter("./logs_sigmoid")
step=1
for data in  dataloader:
    imgs,targets=data
    writer.add_images("input",imgs,step)
    output=zj(imgs)
    writer.add_images("output",output,step)
    step=step+1
writer.close()

