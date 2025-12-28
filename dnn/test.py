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
parameters = sum (p.numel() for n,p in dnn.named_parameters())
print("loss", loss.item(), "grad", dnn.final_layer.linear.weight.grad.abs().mean().item())
print (f"loss : {loss}, field shape : {out.shape}, parameters: {parameters}")

import torch
from dnn import Models

device = "cuda"
dnn = Models["FlowField_XS/4"](temporal_resolution=8, label_dropout=0.0).to(device)
dnn.train()

B,T,C,H,W = 2,8,4,40,40
x = torch.randn(B,T,C,H,W, device=device)
t = torch.rand(B, device=device)
c = torch.zeros(B, device=device, dtype=torch.long)

out = dnn(x,t,c)

target = torch.randn_like(out)
loss = (out - target).pow(2).mean()
loss.backward()

g = dnn.final_layer.linear.weight.grad
print("loss", float(loss), "final_linear_grad_mean", float(g.abs().mean()))