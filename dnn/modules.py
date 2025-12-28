# time step embedder

from itertools import repeat
from typing import Any, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

def modulate(x, shift, scale):
    """
    Modulate signal x (B, aux, T, C)
    with shift/scale (B, C)
    """

    x = x * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
    return x

class FinalLayer(nn.Module):
    """ Final Layer"""
    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.dim = dim
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
    
    def forward(self, x, y):
        scale, shift = self.adaLN_modulation(y).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale) # (B, T, HW, C)
        x = self.linear(x)
        return x

class SelfAttention(nn.Module):
    """
        SelfAttention Module, aggregates information across sequence into individual fragments

        dim: int - dimensionality of the vector space of the neural representation
        n_head: int - number of attention heads, such that dim % n_head == 0
        kqv_bias: bool - whether to use bias for kqv linear projections
    """
    def __init__(self, dim: int, n_head: int=16, kqv_bias=False):
        super().__init__()
        assert dim % n_head == 0, f"dimensionality of the neural representation :{dim} is not divisible by the specified number of attention heads: {n_head}"

        
        self.dim = dim
        self.n_head = n_head
        self.head_size = dim // n_head

        # input should be (B, T, HW, C)
        # or for temporal usecase (B, HW, T, C)

        self.qkv = nn.Linear(dim, 3*dim, bias=kqv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj.RESIDUAL_SKIP = True
    
    def forward(self, x):

        assert x.dim() == 4, f"expected (B, aux, T, C) got {x.shape}"
        # aux_B can be either space (temporal attention) or time (spatial attention)
        B, aux_B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1) # 3x(B, b, T, C)
        q = q.view(B * aux_B, T, self.n_head, self.head_size).transpose(-2, -3) #(B, T, nh, hs) -> (B, nh, T, hs)
        k = k.view(B * aux_B, T, self.n_head, self.head_size).transpose(-2, -3) #(B, T, nh, hs) -> (B, nh, T, hs)
        v = v.view(B * aux_B, T, self.n_head, self.head_size).transpose(-2, -3) #(B, T, nh, hs) -> (B, nh, T, hs)

        # att = (q @ k.transpose(-1,-2)) / math.sqrt(self.head_size)
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        #y = F.scaled_dot_product_attention(q, k, v, is_causal=False) #(B, b, nh, T, hs)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # (B*b_aux, nh, T, hs)
        
        y = y.transpose(-2, -3).contiguous().view(B, aux_B, T, C) # merge heads (B, b, T, C)
        y = self.proj(y)
        return y


class GatedMLP (nn.Module):
    """
        gated MLP with silu activation
        input (B, aux, T, C)
        output (B, aux, T, C)

        args:
        mlp_ratio: float - multiplier to expand dimensionality of vectorspace within the mlp
        bias: bool
        dim: int - vector space dimensionality
    """        
    def __init__(self, dim: int, activation = lambda:nn.SiLU(), mlp_ratio: float = 4.0, bias: bool = True):
        super().__init__()
        self.dim = dim
        self.n_hidden = int (mlp_ratio * dim)
        
        self.fc1 = nn.Linear(dim, 2 * self.n_hidden, bias=bias)
        self.activation = activation()
        self.fc2 = nn.Linear(self.n_hidden, dim, bias=bias)
        self.fc2.RESIDUAL_SKIP = True

    def forward(self, x):

        a, b = self.fc1(x).chunk(2, dim=-1)
        x = self.activation(a) * b
        y = self.fc2(x)
        return y

class LabelEmbedder(nn.Module):
    """
        Embeds class label into vector space R^C
    """

    def __init__(self, num_classes: int, n_embd: int, p_drop: float=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.n_embd = n_embd
        self.p_drop = p_drop    # for CFG setup during training
        self.table = nn.Embedding(num_classes + 1, n_embd) # +1 for null class indexed at num_classes in the table
    
    def forward(self, y: torch.Tensor, force_drop: bool=False):
        """
            y (B,) torch.long or int64
            returns (B,C)
        """
        assert y.dim() == 1, f"expected (B,), got {y.shape}"
        y = y.long()

        if force_drop:
            y = torch.full_like(y, self.num_classes)
        elif self.training and self.p_drop > 0:
            drops = torch.rand(y.shape[0], device=y.device) < self.p_drop
            y = torch.where(drops, torch.full_like(y, self.num_classes), y)
        
        return self.table(y)

class TimeStepEmbedder(nn.Module):
    """
        Embeds scalar continuous time 't' into high dimensional vectorspace using sinusoidal embeddings,
        then refines this vector using MLP to produce n_hidden dimensional representation that is compatible with the 
        DNN
    """

    def __init__(self, n_hidden:int, activation=nn.SiLU, freq_embd:int = 256):
        super().__init__()

        assert freq_embd > 1
        self.freq_embd = freq_embd
        self.n_hidden = n_hidden
        self.activation = activation
        self.mlp = nn.Sequential(
            nn.Linear (freq_embd, n_hidden, bias=True),
            activation(),
            nn.Linear (n_hidden, n_hidden, bias=True)
        )
    
    @staticmethod
    def embed_timestep (t, freq_embd, max_period=10000):
        # half the time embedding dimensions are represented as sines, rest half are represented as cosines.
        # if freq_embd is odd, then append 0 frequence component to maintain consistency

        half = freq_embd // 2
        log_freqs = -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)

        # now we have log_freqs in descending order, they start from 0 and grow negative as we move right
        # later we exponentiate log_freqs to get frequencies, and exp(very large negative number) is very small
        # we dont want frequency components that are really close to 0 and almost indistinguishable from each other at tail end
        # thats why we scale down these negative numbers by "half" so that we have smaller negative numbers

        log_freqs = log_freqs / half # -math.log(max_period) (0....1)
        # note that we resort to exponential scale decay to ensure high range-variance deltas in frequencies
        # df would be constant in linear scale
        # exponentiate to get freqs
        freqs = torch.exp (log_freqs)

        t = t.unsqueeze(-1).float() #(B,1)
        args = t * freqs #(B, half)

        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        if freq_embd % 2 != 0:
            # add 0 frequency component if freq_embd is odd
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, 0]).unsqueeze(-1)), dim=-1)
        return embedding
    
    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
    
    def forward(self, t):
        t_embedding = self.embed_timestep(t, self.freq_embd).to(self.dtype)
        return self.mlp(t_embedding)
    
class TemporalEmbedder(nn.Module):
    """
    Embeds frame idx (T) into (T,C) vector space with sinusiodal embedding
    """
    def __init__(self, n_embd, frame_count, max_period=10000):
        super().__init__()
        assert n_embd != 1
        self.n_embd = n_embd
        self.frame_count = frame_count
        self.max_period = max_period
    
    @staticmethod
    def embed_frame_idx(frame_idx, n_embed, max_period=10000):
        # [0....71] embed this
        log_freqs = - math.log(max_period) * torch.arange(start=0, end=n_embed//2, dtype=torch.float32, device=frame_idx.device) # [,) exclusive at end
        log_freqs = log_freqs / (n_embed // 2)
        freqs = torch.exp(log_freqs)

        frame_idx = frame_idx.unsqueeze(-1) #(frame_count, 1)
        args = frame_idx * freqs # (frame_count, n_embd//2)

        frame_pos_embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1) # (frame_count, n_embd)
        if n_embed % 2 != 0:
            # append a 0 frequence component if n_embd is odd
            frame_pos_embedding = torch.cat((frame_pos_embedding, torch.zeros_like(frame_pos_embedding[:, 0]).unsqueeze(-1)), dim=-1)
        return frame_pos_embedding
    
    def forward (self, frame_idx):
        assert frame_idx.dim() == 1, f"Frame idx tensor must be single dimensional, its : {frame_idx.shape}"
        
        return self.embed_frame_idx(frame_idx, self.n_embd, self.max_period) # (T, C)

def ntuple(n_dim: int, x):
    """
    Converts input into n_dim-tuple. For handling resolutions
    """
    return tuple(repeat(x, n_dim))



def get_2d_sincos_pos_embed(
        n_embed,
        grid_size: Union[int, Tuple[int, int]],
        base_size: int = 16,
        cls_token: bool = False, extra_tokens: int=0,
        pos_inetrp_scale: float=1.0,
):
    """
    One pixel in grid is represented by a vector in R^(n_embd), each half of n_embd dimensions dedicated to representing
    x and y co-ordinate of the pixel
    """

    if isinstance(grid_size, int):
        grid_size = ntuple(2, grid_size)
    
    # interpolate position embeddings to adapt model to different resolutions
    # make it so that specific spatial positions have similar embediigns

    # heigh is dim=0, width is dim=1
    
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / pos_inetrp_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / pos_inetrp_scale

    # width, height
    grid = np.meshgrid(grid_w, grid_h)
    # stack along axis 0 to get two matrices, first for width second for height
    grid = np.stack(grid, axis=0) #(2, W, H)
    # add spurious dimension to be processed by the embedding function
    grid = grid.reshape(2, 1, grid_size[1], grid_size[0]) 

    pos_embedding = get_2d_sinusoidal_embedding_from_grid(n_embed, grid)
    if cls_token and extra_tokens > 0:
        pos_embedding = np.concatenate([np.zeros([extra_tokens, n_embed]), pos_embedding], axis=0)
    
    return pos_embedding # (HW, C)
    

def get_2d_sinusoidal_embedding_from_grid(n_embed, grid):
    "Takes in a grid (2, 1, H, W) returns (HW, C)"

    assert n_embed % 2 == 0
    half_embed = n_embed // 2

    # send x co-ordinates (1, W, H)
    embed_w = get_1d_sinusoidal_embedding(half_embed, grid[0]) # (HW, half_embed)
    # send y co-ordinates 
    embed_h = get_1d_sinusoidal_embedding(half_embed, grid[1]) # (HW, half_embed)
    positional_embedding = np.concatenate([embed_h, embed_w], axis=1)
    return positional_embedding

def get_1d_sinusoidal_embedding(n_embed, grid):
    assert n_embed % 2 == 0

    omega = np.arange(n_embed//2, dtype=np.float64) #(C/2)
    omega = omega / (n_embed // 2) # linearly spaced

    # generate exponentially decaying frequencies
    freqs = (1/10000) ** omega #(C/2)
    grid = grid.reshape(-1) #HW

    mutated_freqs = np.einsum('m,d -> md', grid, freqs) #(HW, D/2)

    sin_embed = np.sin(mutated_freqs)
    cos_embed = np.cos(mutated_freqs)
    return np.concatenate([sin_embed, cos_embed], axis=1)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample/ example level dropout (when applied in main path of residual blocks)
    Kills all activations across all tokens/channels for some random examples in a batch
    """

    def __init__(self, drop_prob: float=0.0, scale_by_keep:bool=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


def drop_path (x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """
    Randomly kill all activations for some examples and scale up survivors to retain same expected value, so we dont have to upscale
    during inference when dropout is disabled

    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)

    This is the same as the DropConnect implementation for EfficientNet, however, the original name is misleading
    as 'Drop Connect' is a different form of droupout in a separate paper.
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    This implementation opts to changing the layer and the argument names to 'drop path' and 'drop_prob' rather than
    DropConnect and survival_rate respectively
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) # (B, T, C) -> (B, 1, 1) so that works with diff dim tensors, not just 2D convnets
    # new tensor same device and dtype as x
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob) # A tensor of shape (B, [1]*(x.ndim-1)) containing 0/1 with keep_prob; broad castable
    
    # if we dont scale_by_keep, during training mask * x scales output down by keep_prob on average
    # to rectify we have to multiply by keep_prob during inference
    # with scaling (inverted dropout) mask = mask / keep_prob, it automatically rectifies by scaling surviving units up by keep_prob
    # so the expected value stays the same during train and test time
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob) 
    return x * random_tensor