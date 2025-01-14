# unet.py conditional 1d unet for diffusion policy
# implementation of conditional 1d unet as described in https://arxiv.org/abs/2303.04137

# this project uses https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py which
# is licensed under the MIT License. The original license can be found at unet_LICENSE.md

# Code has been modified by adding comments, restructuring for clarity and adaptability to the project

import logging
import itertools

import einops.layers
import einops.layers.torch
from torch import nn
import torch
import einops

from typing import Optional

from cyber.models.action.diffusion.backbones.nn_utils import FourierEmb
from cyber.models.action.diffusion.backbones.diffusionbackbone import DiffusionBackbone

logger = logging.getLogger(__name__)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # einops.layers.torch.Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # einops.layers.torch.Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    Conditional Residual Block for 1D data i.e. temporal data.
    Uses FiLM modulation for conditioning.
    """

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8, cond_predict_scale=False):
        """
        args:
            in_channels(int): number of input channels
            out_channels(int): number of output channels
            cond_dim(int): dimension of the conditioning vector
            kernel_size(int): kernel size for the convolutional layers, default=3
            n_groups(int): number of groups for GroupNorm, default=8
            cond_predict_scale(bool): whether to predict scale for FiLM modulation, default=False(only bias is predicted)
        """
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            einops.layers.torch.Rearrange("batch t -> batch t 1"),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(DiffusionBackbone):
    """unet for 1d data with FiLM modulation for conditioning

    Janner et all induced diffusion into robotics policy using a 1d unet https://arxiv.org/abs/2205.09991
    Cheng et al. proposed a conditional unet for diffusion policy https://arxiv.org/abs/2303.04137
    In terms of architecture, this is a conditional unet for 1d data with FiLM modulation for conditioning
    """

    def __init__(
        self,
        input_dim,
        global_cond_dim=None,
        local_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        """
        args:
            input_dim(int): number of input channels
            global_cond_dim(int): dimension of the global conditioning vector
            local_cond_dim(int): dimension of the local conditioning vector, default=None(no local conditioning)
            diffusion_step_embed_dim(int): dimension of the diffusion step embedding, default=256
            down_dims(tuple[int]): dimensions of the unet downsampled layers, default=(256, 512, 1024)
            kernel_size(int): kernel size for the convolutional layers, default=3
            n_groups(int): number of groups for GroupNorm, default=8
            cond_predict_scale(bool): whether to predict scale for FiLM modulation, default=False(only bias is predicted)
        """
        super().__init__()
        all_dims = [input_dim, *list(down_dims)]
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            FourierEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(itertools.pairwise(all_dims))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList(
                [
                    # down encoder
                    ConditionalResidualBlock1D(
                        dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale
                    ),
                    # up encoder
                    ConditionalResidualBlock1D(
                        dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale
                    ),
                ]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale
                ),
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale
                        ),
                        ConditionalResidualBlock1D(
                            dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(self, noise_actions: torch.Tensor, time_step: torch.Tensor, condition: Optional[torch.Tensor] = None, **kwargs):
        """forward pass for the model.

        Since the model is devised for diffusion policy, the argument names are specific to diffusion policy

        args:
            noise_actions(torch.Tensor): input tensor, shape: (batch_size, horizon, input_dim)
            time_step(torch.Tensor): diffusion step
            condition(torch.Tensor): global conditioning tensor, shape: (batch_size, global_cond_dim)

        keyword args:
            local_cond(torch.Tensor): local conditioning tensor, shape: (batch_size, horizon, local_cond_dim) (default=None)

        returns:
            output(torch.Tensor): predicted noise, shape: (batch_size, horizon, input_dim)

        NOTE 1: because of upsampling/downsampling with stride 2, some action horizons may cause shape mismatch.
        In general, avoid using horizons that are not multiples of 2 or too small.

        NOTE 2: the original code has a bug in the upsample path where local features are not added. This has been fixed here.
        """

        sample = einops.rearrange(noise_actions, "b h t -> b t h")  # because conv1d wants batch, channels, length

        # 1. time
        timesteps = time_step
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        if condition is None:
            assert hasattr(self, "global_condition"), "global condition must be provided"
        else:
            self.global_condition = condition
        global_feature = self.diffusion_step_encoder(timesteps)
        global_feature = torch.cat([global_feature, self.global_condition], axis=-1)

        local_cond = kwargs.get("local_cond", None)

        # encode local features
        h_local = []
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, "b h t -> b t h")
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        x = sample
        h = []
        # downsample convolutions
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # single block of mid modules
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # upsample convolutions with skip connections
        for _idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # The original code has a bug here,
            # idx == len(self.up_modules) and len(h_local) > 0 which causes local features to never be added in the upsample path
            # The bug is fixed here and thus makes all original checkpoints incompatible
            x = resnet2(x, global_feature)
            x = upsample(x)
        if len(h_local) > 0:
            x = x + h_local[1]  # implement the fix here
        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")  # back to original shape
        return x
