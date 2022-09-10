import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class NFNeXt(nn.Module):
    def __init__(
        self,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        in_channels: int = 3,
        num_classes: int = 1000,
        num_head_layers: int = 1,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        assert len(depths) == len(dims)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16 * in_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=4, stride=4),
            ScaledStdConv2d(16 * in_channels, dims[0], kernel_size=1),
        )
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).split(depths)
        dp_rates = [x.tolist() for x in dp_rates]
        stages = []
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            stage = []
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
                stage += [Residual(*block)]
            if i < len(depths) - 1:
                stage += [
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    ScaledStdConv2d(dim, dims[i + 1], kernel_size=1),
                ]
            else:
                stage += [LayerNorm2d(dim)]
            stages += [nn.Sequential(*stage)]
        self.stages = nn.Sequential(*stages)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
        )
        if num_head_layers > 0:
            head = []
            for _ in range(num_head_layers - 1):
                head += [nn.Linear(dims[-1], dims[-1]), nn.ReLU(inplace=True)]
            if drop_rate > 0:
                head += [nn.Dropout(drop_rate)]
            self.head = nn.Sequential(
                nn.Sequential(*head), nn.Linear(dims[-1], num_classes)
            )
        self.num_head_layers = num_head_layers
        self.apply(self.init_weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        if self.num_head_layers > 0:
            x = self.head(x)
        return x

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)


class Residual(nn.Sequential):
    # pylint: disable=arguments-renamed
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + super().forward(x)


class Scale2d(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.init_value = init_value
        self.scale = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale[:, None, None] * x

    def extra_repr(self) -> str:
        return f"{self.dim}, init_value={self.init_value}"


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
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


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
        self.gamma = gamma

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

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f", eps={self.eps}, gamma={round(self.gamma, 3):0.3f}"
        )


# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/norm.py


class LayerNorm2d(nn.LayerNorm):
    def __init__(
        self, num_channels: int, eps: float = 1e-6, affine: bool = True
    ) -> None:
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    # pylint: disable=arguments-renamed
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.permute(0, 3, 1, 2)
        return x
