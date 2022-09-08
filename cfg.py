from typing import Any

from torch import nn

from net import NFNeXt


def tiny_1k(**kwargs: Any) -> nn.Module:
    cfg = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), num_classes=1000)
    return NFNeXt(**(cfg | kwargs))


def small_1k(**kwargs: Any) -> nn.Module:
    cfg = dict(
        depths=(3, 3, 27, 3), dims=(96, 192, 384, 768), num_classes=1000
    )
    return NFNeXt(**(cfg | kwargs))
