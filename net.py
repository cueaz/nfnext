import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Scale2d(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.init_value = init_value
        self.scale = nn.Parameter(torch.full((dim, 1, 1), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

    def extra_repr(self) -> str:
        return f"{self.dim}, init_value={self.init_value}"


class NFNeXt(nn.Module):
    def __init__(
        self,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        in_chans: int = 3,
        num_classes: int = 1000,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 16 * in_chans, kernel_size=1),
            nn.AvgPool2d(kernel_size=4, stride=4),
            ScaledStdConv2d(16 * in_chans, dims[0], kernel_size=1),
        )
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).split(depths)
        dp_rates = [x.tolist() for x in dp_rates]
        stages = []
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            stage = []
            if i > 0:
                stage += [
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    ScaledStdConv2d(dims[i - 1], dim, kernel_size=1),
                ]
            for j in range(depth):
                block = [
                    nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
                    ScaledStdConv2d(
                        dim,
                        4 * dim,
                        kernel_size=1,
                        gamma=math.sqrt(2 / (1 - 1 / math.pi)),
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(4 * dim, dim, kernel_size=1),
                    Scale2d(dim),
                ]
                if drop_path_rate > 0:
                    block += [DropPath(dp_rates[i][j])]
                stage += [Residual(nn.Sequential(*block))]
            stages += [nn.Sequential(*stage)]
        self.stages = nn.Sequential(*stages)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.LayerNorm(dims[-1], eps=1e-6),
        )
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(dims[-1], num_classes),
        )
        self.apply(self.init_weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)


# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(
        self, drop_prob: float = 0.0, scale_by_keep: bool = True
    ) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/std_conv.py


class ScaledStdConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        gamma: float = 1.0,
        eps: float = 1e-6,
        gain_init: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.gain = nn.Parameter(
            torch.full((self.out_channels, 1, 1, 1), gain_init)
        )
        self.scale = (
            gamma * self.weight[0].numel() ** -0.5
        )  # gamma * 1 / sqrt(fan-in)
        self.eps = eps

    # pylint: disable=arguments-renamed
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1),
            None,
            None,
            weight=(self.gain * self.scale).view(-1),
            training=True,
            momentum=0.0,
            eps=self.eps,
        ).reshape_as(self.weight)
        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
