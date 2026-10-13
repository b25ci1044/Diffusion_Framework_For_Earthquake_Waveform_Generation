import math,torch
import torch.nn as nn
import torch.nn
from dataclasses import dataclass
import torch.nn.functional as F

from mamaba_ssm import Mamba

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DiffusionModelConfig:
    in_channels: int = 3
    cond_channels: int = 7
    hidden_channels: int = 128
    d_state: int = 128
    d_conv : int = 16
    expand : int = 2

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:

        device = time.device
        half_dim = self.dim // 2

        # f_i = 1/1000^(i/dim_half - 1) Using this fromula to generate scaling frequencies
        scale = math.log(10000) / (half_dim - 1)

        freqs = torch.exp(torch.arange(half_dim, device=device) * -scale)
        args = time[:, None].float() * freqs[None, :]

        return torch.cat([args.sin(), args.cos()], dim=-1)

class ConditionEmbedding(nn.Module):
    def __init__(self,cond_dim: int = 7, emb_dim: int =128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(cond_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)

class ResBlock1D(nn.Module):

    def __init__(self,  in_channels :int , out_channels: int, emb_dim: int):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels * 2)
        )

        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:

        res = self.residual(x)

        h = self.conv1(F.silu(self.norm1(x)))

        scale_shift = self.emb_proj(emb).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)

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
        # Input shape: (Batch, Channels, Seq_Len) -> permute to (Batch, Seq_Len, Channels)
        x_seq = x.permute(0, 2, 1)

        # Forward scan
        out_fwd = self.fwd_mamba(x_seq)

        # Backward scan
        x_rev = torch.flip(x_seq, dims=[1])
        out_bwd = self.bwd_mamba(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])

        # Merge, normalize, and residual add
        merged = self.norm(out_fwd + out_bwd)
        return merged.permute(0, 2, 1) + x


class HybridUMambaDiffusion(nn.Module):
    def __init__(self, config: DiffusionModelConfig):
        super().__init__()
        self.config = config
        emb_dim = config.hidden_dim

        # 1. Conditioning Embeddings
        self.time_embed = SinusoidalTimeEmbedding(emb_dim)
        self.cond_embed = ConditionEmbedding(config.cond_dim, emb_dim)
        self.emb_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # 2. Encoder (Downsampling path)
        self.in_conv = nn.Conv1d(config.in_channels, 64, kernel_size=3, padding=1)
        self.down_block1 = ResBlock1D(64, 64, emb_dim)
        self.down_sample1 = nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=1)

        self.down_block2 = ResBlock1D(64, 128, emb_dim)
        self.down_sample2 = nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1)

        # 3. Bottleneck (Bidirectional Mamba core)
        self.bot_res1 = ResBlock1D(128, 128, emb_dim)
        self.mamba_core = BiMambaBottleneck(
            d_model=128,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand
        )
        self.bot_res2 = ResBlock1D(128, 128, emb_dim)

        # 4. Decoder (Upsampling path with Skip Connections)
        self.up_sample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_block2 = ResBlock1D(128 + 128, 64, emb_dim)

        self.up_sample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_block1 = ResBlock1D(64 + 64, 64, emb_dim)

        # 5. Denoising Output Head
        self.out_norm = nn.GroupNorm(8, 64)
        self.out_conv = nn.Conv1d(64, config.in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Fuse time step and metadata condition vectors
        t_emb = self.time_embed(time)
        c_emb = self.cond_embed(cond)
        emb = self.emb_mlp(t_emb + c_emb)

        # Stem & Encoder
        h = self.in_conv(x)
        skip1 = self.down_block1(h, emb)
        h = self.down_sample1(skip1)

        skip2 = self.down_block2(h, emb)
        h = self.down_sample2(skip2)

        # Bottleneck
        h = self.bot_res1(h, emb)
        h = self.mamba_core(h)
        h = self.bot_res2(h, emb)

        # Decoder with Skip Connections & sequence-length alignment
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

        # Prediction
        return self.out_conv(F.silu(self.out_norm(h)))




model = HybridUMambaDiffusion(DiffusionModelConfig(DiffusionModelConfig())).to(device)

batch_size = 4
sequence_length = 256 # Can be any length, but typically powers of 2 (128, 256, 512)

# x: The noisy input sequence. Shape: (Batch, Channels, Sequence_Length)
dummy_x = torch.randn(batch_size, DiffusionModelConfig().in_channels, sequence_length).to(device)

# time: The diffusion timesteps. Shape: (Batch,)
# Usually random integers between 0 and your max diffusion steps (e.g., 1000)
dummy_time = torch.randint(0, 1000, (batch_size,)).to(device)

# cond: The conditioning metadata. Shape: (Batch, Cond_Dim)
dummy_cond = torch.randn(batch_size, DiffusionModelConfig().cond_dim).to(device)

# 4. Forward pass
output = model(dummy_x, dummy_time, dummy_cond)


