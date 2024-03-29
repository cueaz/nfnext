# pylint: disable=wrong-import-position
dependencies = ["torch"]

from functools import partial
from typing import Any

from torch import nn
from torch.hub import load_state_dict_from_url

import cfg

_BASE_URL = "https://github.com/cueaz/nfnext/releases/download"
_MODEL_URLS = {
    "tiny_1k": {"imagenet_220908": f"{_BASE_URL}/"},
    "small_1k": {"imagenet_220908": f"{_BASE_URL}/"},
}


def _model(
    name: str,
    pretrained: str | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    model = getattr(cfg, name)(**kwargs)
    if pretrained is not None:
        state_dict = load_state_dict_from_url(
            _MODEL_URLS[name][pretrained],
            progress=progress,
            check_hash=True,
            map_location="cpu",
        )
        model.load_state_dict(state_dict)
    return model


tiny_1k = partial(_model, "tiny_1k")
small_1k = partial(_model, "small_1k")
