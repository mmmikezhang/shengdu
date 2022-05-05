import torch
from model_save import *

model=torch.load('zj_method.pth')
print(model)
zj=Zj()
zj.load_state_dict(torch.load('zj_method.pth'))
print(zj)