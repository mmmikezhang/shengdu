from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# E:\shengdustudy\learn\dataset\train\ants_image\0013035.jpg
img_path="E:/shengdustudy/learn/dataset/train/ants_image/0013035.jpg"
img=Image.open(img_path)   #img里存放的是picture(pic类型)的数据
# print(img)
writer=SummaryWriter("logs")
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)          #按alt加p可以看需要什么参数
# pic类型转换成tensor类型
# print(tensor_img)
writer.add_image("Tensor_img",tensor_img)
writer.close()
