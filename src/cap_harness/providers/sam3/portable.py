"""Run the SAM3 service on hosts without CUDA or triton, e.g. Apple silicon on CPU.

SAM3's image path hardcodes CUDA in a few tensor factories, and it imports video and
training modules that need triton and decord. These shims apply only inside this
service process: CUDA factory requests go to the selected device, pinning is a no-op,
autocast is disabled (bfloat16 on this CPU took minutes per image instead of seconds),
and the two video-only modules raise if anything actually calls them.

    python -m cap_harness.providers.sam3.portable --device cpu --port 8114

Measured on an Apple M5 Pro: about 5 s per image and prompt on CPU. MPS fails inside
SAM3's geometry encoder, so CPU is the supported device here.
"""

from __future__ import annotations

import contextlib
import sys
import types

import torch

_FACTORIES = ("zeros", "ones", "arange", "empty", "full", "tensor", "linspace", "randn", "rand")


def _unavailable(*_args, **_kwargs):
    raise RuntimeError("This SAM3 feature needs triton or decord, which this host lacks")


class _NoAutocast(contextlib.ContextDecorator):
    """Stands in for torch.autocast both as a context manager and as a decorator."""

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _redirecting(factory, device):
    def redirect(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = device
        return factory(*args, **kwargs)

    return redirect


def install_shims(device: str) -> None:
    edt = types.ModuleType("sam3.model.edt")
    edt.edt_triton = _unavailable
    decord = types.ModuleType("decord")
    decord.cpu = decord.VideoReader = _unavailable
    sys.modules.setdefault("sam3.model.edt", edt)
    sys.modules.setdefault("decord", decord)
    for name in _FACTORIES:
        setattr(torch, name, _redirecting(getattr(torch, name), device))
    torch.Tensor.cuda = lambda self, *_args, **_kwargs: self.to(device)
    torch.Tensor.pin_memory = lambda self, *_args, **_kwargs: self
    torch.autocast = _NoAutocast


def main() -> None:
    if "--device" not in sys.argv:
        sys.argv += ["--device", "cpu"]
    install_shims(sys.argv[sys.argv.index("--device") + 1])
    from .service import main as serve

    serve()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
