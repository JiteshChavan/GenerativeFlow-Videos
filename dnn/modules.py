# time step embedder

from itertools import repeat
from typing import Any, Optional, Union, Tuple

import torch
import torch.nn as nn
import math
import numpy as np

class TimeStepEmbedder(nn.Module):
    """
        Embeds scalar continuous time 't' into high dimensional vectorspace using sinusoidal embeddings,
        then refines this vector using MLP to produce n_embd dimensional representation that is compatible with the 
        DNN
    """

    def __init__(self, freq_embd:int, n_hidden:int, activation):
        super().__init__()
        self.freq_embd = freq_embd
        self.n_hidden = n_hidden
        self.activation = activation
        self.mlp = nn.Sequential(
            nn.Linear (freq_embd, n_hidden, bias=True),
            self.activation,
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
        assert frame_idx.shape[0] == self.frame_count, f"frame_count:{self.frame_count} and frame_idx:{frame_idx.shape} shape mismatch"
        
        return self.embed_frame_idx(frame_idx, self.n_embd) # (T, C)

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

