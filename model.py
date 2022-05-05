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


# 测试
if __name__ == '__main__':
    zj=Zj()
    input=torch.ones((64,3,32,32))
    output=zj(input)
    print(output)
    print(output.shape)
