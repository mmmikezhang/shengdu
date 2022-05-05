from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer=SummaryWriter("logs")
img=Image.open("E:/shengdustudy/learn/dataset/train/ants_image/0013035.jpg")
print(img)

# 1、ToTensor
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("totensor",img_tensor)


#2、Normalize正则化
print(img_tensor[0][0][0])
# 三维向量要其中第一行，第一例，第一。。
trans_norm=transforms.Normalize([8,8,1],[3,2,9])
img_norm=trans_norm(img_tensor)
print(img_norm[0][0][0])

writer.add_image("Normalize",img_norm,2)
# 添加到tensorboard中进行显示
# 2代表step



# 3、Resize
print(img.size)
trans_resize=transforms.Resize(512,512)
img_resize=trans_resize(img)
img_resize=trans_totensor(img_resize)
print(img_resize)

# RandomCrop
trans_rand

writer.close()


