import torchvision
from PIL import Image
import torch
from torch import nn

image_path="E:\shengdustudy\imgs\dog1.jpg"
image=Image.open(image_path)  #读取成PIL形式的图片
# print(image)

transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])

image=transform(image)
# print(image)


# 搭建神经网络



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


model=torch.load('zj_29.pth')
# print(model)

# print(model)
zj=Zj()
image=torch.reshape(image,(-1,3,32,32),)

# print(image.size())
model.eval()
with torch.no_grad():

    output=zj(image)


print(output)

print(output.argmax(1))
