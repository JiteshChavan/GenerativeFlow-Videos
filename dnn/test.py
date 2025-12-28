from dnn import Models
import torch


dnn = Models["FlowField_XS/4"]()

B, T, C, H, W = 2, 8, 4, 40, 40
x = torch.randn(B, T, C, H, W).to('cuda')
t = torch.ones(B).to('cuda') * 0.5
c = torch.zeros(B).to('cuda')

dnn.to('cuda')

out = dnn(x, t, c)
loss = (out**2).mean()
loss.backward()
print (f"loss : {loss}, field shape : {out.shape}")