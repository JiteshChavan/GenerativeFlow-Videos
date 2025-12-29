import torch
import torch.nn.functional as F
import torch.nn as nn


from modules import SelfAttention, FinalLayer, TimeStepEmbedder,TemporalEmbedder, LabelEmbedder, GatedMLP, DropPath, modulate
from modules import get_2d_sincos_pos_embed
from timm.layers import PatchEmbed

import math

class SpaceTimeBlock(nn.Module):
    """
        A simple neural network block that implements factorized space time attention over video signals in R^d by first aggregating information across all pixels within frames with self attention;
        followed by aggregating temporal information for pixels across all frames again with self attention mechanism.


        args:
            dim : int - dimensionality of the neural vectorspace for each fragment of input signal
            y_dim : int - dimensionality of f(class_label, flow_time) conditioning vector
            norm_cls : norm layer type (LayerNorm)
            drop_path : dropout prob

            x : (B, T, HW, C) - input video signal in R^d
            y : (B, C) - aggregation of cts time variable and neural vectorspace representation of class conditioning for approximating the flow field
    """

    def __init__(self, dim: int, y_dim=None, norm_cls=nn.LayerNorm, drop_path=0.0, use_temporal_attention=True):
        super().__init__()
        self.dim = dim
        y_dim = dim if y_dim is None else y_dim

        self.spatial_att = SelfAttention(dim, kqv_bias=True)
        self.norm_s = norm_cls(dim)
        if use_temporal_attention:
            self.temporal_att = SelfAttention(dim, kqv_bias=True)
            self.norm_t = norm_cls(dim)
        
        self.norm_mlp = norm_cls(dim)
        mlp_activation = lambda: nn.SiLU()
        self.mlp = GatedMLP(dim, activation=mlp_activation, bias=True)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(y_dim, 9*dim if use_temporal_attention else 6*dim, bias=True))
        

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_temporal_attention = use_temporal_attention
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Forward
        
        args:
        x : tensor (B, T, HW, C) representation from previous block
        y : tensor (B, C) f(flow_time, class_label) conditioning vector

        returns x
        """

        # do adaLN(spatial_attention(x)) then adaLN(temporal_attention(x)) then adaLN(temporal_MLP(x))

        if self.use_temporal_attention:
            
            scale_space_msa, gate_space_msa, shift_space_msa, scale_time_msa, gate_time_msa, shift_time_msa, scale_mlp, gate_mlp, shift_mlp = self.adaLN_modulation(y).chunk(9, dim=-1)
            # x (B, T, HW, C)
            x = x +  self.drop_path(gate_space_msa[:,None, None, :] * self.spatial_att(modulate(self.norm_s(x), shift_space_msa, scale_space_msa))) # (B, T, HW, C) information aggregation across all HW
            # prime the signal for temporal attention
            x = x.transpose(1, 2) # x (B, HW, T, C)
            x = x + self.drop_path(gate_time_msa[:, None, None, :] * self.temporal_att(modulate(self.norm_t(x), shift_time_msa, scale_time_msa))) # (B, HW, T, C)
            # restructure signal in original format for next block 
            x = x.transpose(1, 2) # x (B, T, HW, C)
            # mlp
            x = x + self.drop_path(gate_mlp[:, None, None, :] * self.mlp(modulate(self.norm_mlp(x), shift_mlp, scale_mlp))) # (B, T, HW, C)

        else:
            scale_space_msa, gate_space_msa, shift_space_msa, scale_mlp, gate_mlp, shift_mlp = self.adaLN_modulation(y).chunk(6, dim=-1)
            # x (B, T, HW, C)
            x = x +  self.drop_path(gate_space_msa[:,None, None, :] * self.spatial_att(modulate(self.norm_s(x), shift_space_msa, scale_space_msa))) # (B, T, HW, C) information aggregation across all HW
            # mlp
            x = x + self.drop_path(gate_mlp[:, None, None, :] * self.mlp(modulate(self.norm_mlp(x), shift_mlp, scale_mlp))) # (B, T, HW, C)

        return x


class DNN(nn.Module):
    """
        Deep Neural Network backbone to approximate flow field for generative flow model

    """

    def __init__(self,
            in_channels: int = 4, out_channels: int = 4, spatial_resolution: int = 40, temporal_resolution: int = 48,
            patch_size: int = 4, dim: int = 256, depth: int = 6, learnable_pe: bool = False, label_dropout: float = 0.1,
            drop_path: float = 0.0,
            num_classes: int = 12,

            use_temporal_attention: bool = True,
        ):
        super().__init__()

        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dim = dim
        self.depth = depth

        self.learnable_pe = learnable_pe
        self.label_dropout = label_dropout

        self.num_classes = num_classes

        self.use_temporal_attention = use_temporal_attention

        # B, T, HW, C embedder
        self.x_embedder = PatchEmbed(spatial_resolution, patch_size=patch_size, in_chans=in_channels, embed_dim=dim)
        num_patches = self.x_embedder.num_patches

        self.t_embedder = TimeStepEmbedder(dim)
        self.temporal_embedder = TemporalEmbedder(dim, frame_count=temporal_resolution)
        self.class_embedder = LabelEmbedder(num_classes, dim, label_dropout)

        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches, dim), requires_grad=learnable_pe) # (B, T, HW, C)

        grid_size = int(math.sqrt(num_patches))


        # drop path setup
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.depth)] # stochastic depth decay rule
        inter_dpr = dpr # drop path value for blocks across depth

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.blocks = nn.ModuleList([
            SpaceTimeBlock(
                dim,
                drop_path=inter_dpr[i],
                use_temporal_attention=use_temporal_attention
            ) for i in range(self.depth)
        ])

        self.final_layer = FinalLayer(dim, patch_size, out_channels=out_channels)
        self.initialize_weights()

    
    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        grid_size = int(math.sqrt(self.x_embedder.num_patches))
        assert grid_size * grid_size == self.x_embedder.num_patches, f"self.x_embededr.num_patches:{self.x_embedder.num_patches} is not a perfect square \n this breaks generation of sincos embeddings under current implementation"
        pos_embed = get_2d_sincos_pos_embed(self.dim, grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0).unsqueeze(0)) #(1, 1, HW, C)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.reshape(w.shape[0], -1)) # reshape incase w is not contiguous

        self.apply(self._init_weights)

        # zero-out adaLN modulation layers in blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
    def _init_weights (self, module):
        if isinstance (module, nn.Linear):
            std = 0.02
            if hasattr (module, 'RESIDUAL_SKIP') and module.RESIDUAL_SKIP:
                if self.use_temporal_attention:
                    std *= (3 * self.depth) ** -0.5 # 3 skips within 1 block
                else:
                    std *= (2 * self.depth) ** -0.5 # 2 skips within 1 block
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance (module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, x, t, c):
        """ 
            x (B, T, C, H, W)
            t (B)
            c (B)
        """
        
        B, T, C, H, W = x.shape

        x = x.reshape(B*T, C, H, W) # reshape in case x from data loader is not contiguous
        x = self.x_embedder(x).view(B, T, self.x_embedder.num_patches, self.dim) # (BT, HW, C) -> (B, T, HW, C)
        x = x + self.pos_embed #(B, T, HW, C) + (1, 1, HW, C)

        frame_idx = torch.arange(T, dtype=torch.float32, device=x.device)
        temporal_embedding = self.temporal_embedder(frame_idx) # (T, C)
        x = x + temporal_embedding[None, :, None, :] # (B, T, HW, C)

        t = self.t_embedder(t) #(B, C)
        c = self.class_embedder(c) #(B, C)


        y = t + c # (B, C)

        for block in self.blocks:
            x = block(x, y) # (B, T, HW, C)
        
        x = self.final_layer(x, y) # (B, T, HW, pqC)
        x = self.unpatchify(x) # (B, T, C, H, W)
        return x
    
    def forward_with_cfg(self, x, t, c, cfg_scale=1.0):
        """
        Classifier Free guidance for flow ODE sampling
        
        x : (B, T, C, H, W)
        t : (B)
        c : (B)
        cfg_scale : strength of guidance
        """
        assert (c >= 0).all() and (c < self.class_embedder.num_classes).all(), "c must be in [0, num_classes-1] for conditional pass"
        assert not self.training

        B = x.shape[0]
        x = torch.cat((x,x), dim=0) #(2B, T, C, H, W)
        t = torch.cat((t,t), dim=0) #(2B)

        c_uncond = torch.full_like(c, self.class_embedder.num_classes)
        c = torch.cat((c, c_uncond), dim=0) #(2B)

        vf = self.forward(x, t, c)
        vf_cond, vf_uncond = vf[:B], vf[B:]
        cfg_vf = vf_uncond + cfg_scale * (vf_cond - vf_uncond)
        return cfg_vf
    

    def unpatchify(self, x):
        """
        input x (B, T, HW, p**2 * C)
        output x (B, T, C, H, W)
        """
        C = self.out_channels
        p = self.patch_size
        h = w = int (x.shape[2] ** 0.5)
        assert h * w == x.shape[2], f"num patches:{x.shape[2]} is not a prefect square hence h/w setup fails for unpatchify"

        x = x.reshape(x.shape[0], x.shape[1], h, w, p, p, C)
        x = torch.einsum('BThwpqC -> BTChpwq', x)
        x = x.reshape(x.shape[0], x.shape[1], C, h*p, w*p)
        return x


def flowField_xs4(**kwargs):
    return DNN(
        depth=6,
        dim=256,
        patch_size=4,
        **kwargs,
    )

def flowField_s4(**kwargs):
    return DNN(
        depth=8,
        dim=512,
        patch_size=4,
        **kwargs,
    )

def flowField_s2(**kwargs):
    return DNN(
        depth=8,
        dim=384,
        patch_size=2,
        **kwargs,
    )

Models = {
    "FlowField_XS/4" : flowField_xs4,
    "FlowField_S/4" : flowField_s4,
    "FlowField_S/2" : flowField_s2,
}

def create_dnn(config):
    assert config.dnn_spec in Models.keys(), f"Invalid dnn_spec: {config.dnn_spec}"
    return Models[config.dnn_spec](
            in_channels = config.latent_channels,
            out_channels = config.latent_channels, 
            spatial_resolution=config.latent_res, 
            temporal_resolution = config.temporal_res,
            learnable_pe = config.learnable_pe,
            label_dropout = config.label_dropout,
            drop_path = config.drop_path,
            num_classes = config.num_classes,
            use_temporal_attention = config.use_temporal_attention
    )