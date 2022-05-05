from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer=SummaryWriter("logs")
image_path="E:/shengdustudy/learn/dataset/train/ants_image/0013035.jpg"
# learn\dataset\train\ants_image\0013035.jpg相对路径报错了，改成了绝对路径
# E:\shengdustudy\learn\dataset\train\ants_image\0013035.jpg
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)
print(type(img_array))
# 变量转换，把PIL类型转换为numpy类型
print(img_array.shape)
writer.add_image("test",img_array,1,dataformats='HWC')
# Args:
# tag(string): Data identifier
# img_tensor(torch.Tensor, numpy.array, or string / blobname): Imagedata
# global_step(int): Global step value to record
# walltime(float): Optional override default
# walltime(time.time())
# seconds
# after
# epoch
# of
# event
for i in range(100):
    # 0-99
    writer.add_scalar("y=2x",2*i,i)
#     scalar_value  y轴
#     global_step    x轴
writer.close()
