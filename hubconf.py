# pylint: disable=wrong-import-position
dependencies = ["torch"]

from functools import partial
from typing import Any

from torch import nn
from torch.hub import load_state_dict_from_url

import cfg

_BASE_URL = "https://github.com/cueaz/nfnext/releases/download"
_model_urls = {
    "tiny": {"i1k_220908": f"{_BASE_URL}/"},
    "small": {"i1k_220908": f"{_BASE_URL}/"},
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
            _model_urls[name][pretrained],
            progress=progress,
        )
        model.load_state_dict(state_dict)
    return model


tiny = partial(_model, "tiny")
small = partial(_model, "small")
