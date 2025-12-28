import torch
import torch.nn.functional as F


class FlowMatching:
    """
        Conditional Flow Matching abstraction
    """
    def __init__(self, dnn, t_sampler="uniform"):
        self.dnn = dnn
        self.t_sampler = t_sampler
    
    def sample_t(self, B, device):
        assert self.t_sampler in ["uniform", "logit_normal"], f"specified time sampler : {self.t_sampler} is not implemented."
        if self.t_sampler == "uniform":
            t = torch.rand(B, device=device)
        elif self.t_sampler == "logit_normal":
            # concentrates more near 0/1 depending on sigma
            sigma = 1.0
            t = torch.randn(B, device=device)*sigma
            t = torch.sigmoid(t)
        return t

    def interp(self, z, eps, t):
        t = t.view(t.shape[0], 1, 1, 1, 1) # (B, T, C, H, W)
        x_t = t * z + (1 - t) * eps # simple condOT transport for now
        v_target = z - eps
        return x_t, v_target
    
    def training_step(self, z, c):
        """
        (z, c) ~ p_data(z,c) z in (B, T, C, H, W) c in (B)
        """

        device = z.device
        B = z.shape[0]

        eps = torch.randn_like(z)
        t = self.sample_t(B, device)
        x_t, u_target = self.interp(z, eps, t)

        u_theta = self.dnn(x_t, t, c)

        loss = ((u_theta - u_target)**2).mean()
        return loss 