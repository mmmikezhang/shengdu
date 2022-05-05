import  torch
from torch.nn import L1Loss
from torch import nn
inputs=torch.tensor([1,2,3],dtype=torch.float32 )
targets=torch.tensor([1,2,5],dtype=torch.float32)

inputs=torch.reshape(inputs,(1,1,1,3))
targets=torch.reshape(targets,(1,1,1,3))

loss=L1Loss()
loss1=nn.MSELoss()
result=loss(inputs,targets)
result2=loss1(inputs,targets)
print(result)
print(result2)

