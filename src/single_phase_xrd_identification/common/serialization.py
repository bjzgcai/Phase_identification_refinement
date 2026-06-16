from __future__ import annotations

import warnings
from os import PathLike
from typing import Any

import torch


def safe_torch_load(
    path: str | PathLike[str],
    map_location: Any = None,
    *,
    trusted: bool = False,
) -> Any:
    """Load a PyTorch artifact with the safest available deserialization mode.

    Public verification paths should use the default ``trusted=False`` so that
    PyTorch 2.x uses ``weights_only=True``. Training resume checkpoints may set
    ``trusted=True`` for local, self-generated files that need optimizer/scaler
    state and therefore may require the legacy pickle loader.
    """
    kwargs = {"map_location": map_location}
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except TypeError:
        warnings.warn(
            "This PyTorch version does not support weights_only=True; "
            "falling back to the legacy torch.load path. Only load trusted checkpoints.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, **kwargs)
    except Exception as exc:
        if not trusted:
            raise RuntimeError(
                f"Could not safely load checkpoint {path!s} with weights_only=True. "
                "Only use checkpoints from trusted sources, or regenerate the file locally."
            ) from exc
        warnings.warn(
            f"Falling back to legacy torch.load for trusted local checkpoint {path!s}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, weights_only=False, **kwargs)
