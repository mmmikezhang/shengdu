from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("logs")

for i in range(100):
    # 0-99
    writer.add_scalar("y=2x",2*i,i)
#     scalar_value  y轴
#     global_step    x轴
writer.close()
