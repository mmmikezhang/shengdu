import torchvision
# 准备的测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

testdata=torchvision.datasets.CIFAR10("./data",False,transform=torchvision.transforms.ToTensor())

test_loader=DataLoader(dataset=testdata,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
# 测试数据集中第一突破一级target
img,target=testdata[0]
print(img.shape)
print(target)
writer=SummaryWriter("dataloader1")

for epoch in range(2):
    step=0
    for data in test_loader:
        imgs,targets=data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step=step+1
    writer.close()
