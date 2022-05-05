import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),

])
trainset=torchvision.datasets.CIFAR10(root="./data",train=True,transform=dataset_transform,download=True)
testset=torchvision.datasets.CIFAR10(root="./data",train=True,transform=dataset_transform,download=True)
# transform=dataset_transform代码要把picture类型的数据转化为tensor类型


# print(testset[0])
# print(testset.classes)
#
# img,target=testset[0]
# print(img)
# print(target)
# print(testset.classes[target])
# img.show()
print(testset[0])
writer=SummaryWriter("p10")
for i in range(10):
    img,target=testset[i]
    writer.add_image("test",img,i)
writer.close()



# 查看tensorboard代码
#tensorboard --logdir=learn\p10
