import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


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
             Linear(in_features=64, out_features=10)  # 因为是

        )


    def forward(self,x):
       x=self.model1(x)



       return x


zj=Zj()
print(zj)
input=torch.ones((64,3,32,32))
output=zj(input)
print(output.shape)

writer=SummaryWriter("logs_seq")
writer.add_graph(zj,input)
writer.close()