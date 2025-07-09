import torch
from torch import nn
import torch.nn.functional as F


class FourierUnit(nn.Module):
    """Fourier Unit for frequency domain feature fusion."""#FDFM

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

        self.conv_spatial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_feat = self.conv_spatial(x)

        # FFT
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        # Frequency domain convolution
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        # IFFT
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:])
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output + spatial_feat


class SpatialAttention(nn.Module):
    """Spatial attention module."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial_net = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.spatial_net(x)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.size()[:2]
        attention = self.avg_pool(x).view(b, c)
        attention = self.fc(attention).view(b, c, 1, 1)
        return x * attention.expand_as(x)


class ResidualBranch(nn.Module):
    """Residual branch with multi-scale feature extraction.""" #RPB

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            ) for dilation in [1, 3, 5]
        ])
        self.spatial_attention = SpatialAttention(channels)
        self.channel_attention = ChannelAttention(channels)
        self.fusion = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale_feats = [conv(x) for conv in self.dilated_convs]
        att = self.spatial_attention(self.channel_attention(x))

        enhanced_feats = [att  for att  in multi_scale_feats]
        return self.fusion(torch.cat(enhanced_feats, dim=1))


class CrossPromotionModule(nn.Module):
    """Cross promotion module for feature interaction."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial_to_freq = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.freq_to_spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor) -> tuple:
        freq_enhanced = freq_feat * self.spatial_to_freq(spatial_feat)
        spatial_enhanced =  self.freq_to_spatial(freq_feat)
        return spatial_enhanced, freq_enhanced


class IterationBlock(nn.Module):
    """Single iteration block combining frequency and residual branches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fourier_branch = FourierUnit(channels, channels)
        self.residual_branch = ResidualBranch(channels)
        self.cross_promotion = CrossPromotionModule(channels)

    def forward(self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor) -> tuple:
        spatial_enhanced, freq_enhanced = self.cross_promotion(spatial_feat, freq_feat)
        spatial_out = self.residual_branch(spatial_enhanced + spatial_feat)
        freq_out = self.fourier_branch(freq_enhanced + freq_feat)
        return spatial_out, freq_out


class FusionNetwork(nn.Module):
    """Main fusion network."""

    def __init__(self, init_channels: int = 32, num_iterations: int = 3) -> None:
        super().__init__()
        self.num_iterations = num_iterations

        # Initial feature extraction
        self.init_spatial = nn.Sequential(
            nn.Conv2d(1, init_channels, 3, 1, 1),
            nn.BatchNorm2d(init_channels),
            nn.ReLU()
        )
        self.init_freq = nn.Sequential(
            nn.Conv2d(2, init_channels, 3, 1, 1),
            nn.BatchNorm2d(init_channels),
            nn.ReLU()
        )

        # Iteration blocks
        self.iteration_block = IterationBlock(init_channels)

        # Fusion decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(init_channels, init_channels, 3, 1, 1),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(),
            nn.Conv2d(init_channels, init_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(init_channels // 2),
            nn.ReLU(),
            nn.Conv2d(init_channels // 2, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, visible: torch.Tensor, infrared: torch.Tensor) -> tuple:
        fea_map=[]
        # Initial feature extraction
        difference = (infrared - visible).detach()
        spatial_feat = self.init_spatial(difference)
        freq_feat = self.init_freq(torch.cat([visible, infrared], dim=1))

        fea_map.append([spatial_feat,freq_feat])

        # Iterative refinement
        for _ in range(self.num_iterations):
            spatial_feat, freq_feat = self.iteration_block(spatial_feat, freq_feat)
            fea_map.append([spatial_feat, freq_feat])

        # Generate fusion result
        fusion_feat = spatial_feat + freq_feat
        fused_image = self.decoder(fusion_feat)
        fea_map.append([fusion_feat])

        return fused_image, spatial_feat,fea_map


class AuxiliaryDecoder(nn.Module):
    """Auxiliary decoder for difference map reconstruction."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels // 2, 3, 1, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


def normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0,1] range."""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)