import torch
from torch import nn


class Zj(nn.Module):
    def __init__(self):
        super(Zj, self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=3)
    def forward(self,x):
        x=self.conv1(x)
        return x
zj=Zj()
torch.save(zj.state_dict(),"zj_method.pth")