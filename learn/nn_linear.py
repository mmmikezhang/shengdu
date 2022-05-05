import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader


dataset=torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor() )
dataloader=DataLoader(dataset,batch_size=64)

class Zj(nn.Module):
    def __init__(self):
        super(Zj, self).__init__()
        self.linear1=Linear(in_features=196608,out_features=10)
    def forward(self,input):
        output=self.linear1(input)
        return output
zj=Zj()



for data in  dataloader:
    imgs,targets=data
    print(imgs.shape)
    output=torch.flatten(imgs)
    print(output.shape)
    output=zj(output)
    print(output.shape)
