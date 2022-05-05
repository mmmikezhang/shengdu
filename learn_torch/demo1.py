from torch.utils.data import Dataset
from PIL import Image
import os
class MyData(Dataset):
    # 类Mydata继承Dataset类
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        # 把root_dir="learn_torch\dataset\train"和label_dir="ants"俩个路径合起来
        self.img_path=os.listdir(self.path)
    # self.img_path里面是每一个图片的名字
    # root_dir文件夹下所有的图片
    def __getitem__(self, idx):

        img_name=self.img_path[idx]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

root_dir="learn_torch/dataset/train"
# learn_torch\dataset\train
ants_label_dir="ants"
bees_label_dir="bees"

ants_dataset=MyData(root_dir,ants_label_dir)
bees_dataset=MyData(root_dir,bees_label_dir)
# 用类MyData来创建实例ants_dataset   其中包含 img_path ，path，label_dir，root_dir







