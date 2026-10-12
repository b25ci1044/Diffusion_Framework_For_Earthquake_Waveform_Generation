import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


@dataclass
class DiffusionModelConfig:
    in_channels: int = 3
    hidden_dim: int = 128
    cond_dim: int = 3
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class ConditionEmbedding(nn.Module):
    """Projects continuous physical earthquake metadata into embedding space."""
    def __init__(self, cond_dim: int, emb_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.proj(cond)

class ResBlock1D(nn.Module):
    """
    1D Residual Block with Adaptive Group Normalization (Scale-Shift Modulation).
    """
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.cond_proj = nn.Linear(emb_dim, out_channels * 2)

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)

        h = self.conv1(F.silu(self.norm1(x)))

        scale_shift = self.cond_proj(F.silu(emb))
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale[:, :, None]
        shift = shift[:, :, None]

        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(F.silu(h))

        return h + res

class BiMambaBottleneck(nn.Module):
    """
    Bidirectional Mamba processing the sequence in forward and reverse 
    temporal directions simultaneously to prevent causal bias in denoising.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba-ssm is not installed. Install via `pip install mamba-ssm causal-conv1d`.")
        
        self.fwd_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.permute(0, 2, 1)

        out_fwd = self.fwd_mamba(x_seq)

        x_rev = torch.flip(x_seq, dims=[1])
        out_bwd = self.bwd_mamba(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])

        merged = self.norm(out_fwd + out_bwd)
        return merged.permute(0, 2, 1) + x

class HybridUMambaDiffusion(nn.Module):
    """
    Conditional Hybrid U-Mamba architecture for 3-component seismic waveform diffusion.
    """
    def __init__(self, config: DiffusionModelConfig):
        super().__init__()
        self.config = config
        emb_dim = config.hidden_dim

        self.time_embed = SinusoidalTimeEmbedding(emb_dim)
        self.cond_embed = ConditionEmbedding(config.cond_dim, emb_dim)
        self.emb_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.in_conv = nn.Conv1d(config.in_channels, 64, kernel_size=3, padding=1)
        self.down_block1 = ResBlock1D(64, 64, emb_dim)
        self.down_sample1 = nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=1)

        self.down_block2 = ResBlock1D(64, 128, emb_dim)
        self.down_sample2 = nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1)

        self.bot_res1 = ResBlock1D(128, 128, emb_dim)
        self.mamba_core = BiMambaBottleneck(
            d_model=128,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand
        )
        self.bot_res2 = ResBlock1D(128, 128, emb_dim)

        self.up_sample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_block2 = ResBlock1D(128 + 128, 64, emb_dim)

        self.up_sample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_block1 = ResBlock1D(64 + 64, 64, emb_dim)

        self.out_norm = nn.GroupNorm(8, 64)
        self.out_conv = nn.Conv1d(64, config.in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(time)
        c_emb = self.cond_embed(cond)
        emb = self.emb_mlp(t_emb + c_emb)

        h = self.in_conv(x)
        skip1 = self.down_block1(h, emb)
        h = self.down_sample1(skip1)

        skip2 = self.down_block2(h, emb)
        h = self.down_sample2(skip2)

        # Bottleneck
        h = self.bot_res1(h, emb)
        h = self.mamba_core(h)
        h = self.bot_res2(h, emb)

        h = self.up_sample2(h)
        if h.shape[-1] != skip2.shape[-1]:
            h = F.pad(h, (0, skip2.shape[-1] - h.shape[-1]))
        h = torch.cat([h, skip2], dim=1)
        h = self.up_block2(h, emb)

        h = self.up_sample1(h)
        if h.shape[-1] != skip1.shape[-1]:
            h = F.pad(h, (0, skip1.shape[-1] - h.shape[-1]))
        h = torch.cat([h, skip1], dim=1)
        h = self.up_block1(h, emb)

        # Predict
        return self.out_conv(F.silu(self.out_norm(h)))