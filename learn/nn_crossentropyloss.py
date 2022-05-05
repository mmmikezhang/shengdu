import torch
import torch.nn
x=torch.tensor([0.1,0.2,0.3])
y=torch.tensor([1])

x=torch.reshape(x,(1,3))
# 1个bachsize  3类
loss_cross= torch.nn.CrossEntropyLoss()
result=loss_cross(x,y)
print(result)