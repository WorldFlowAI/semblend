"""Monkey-patch SGLang's RadixCache with SemBlend semantic donor discovery.

Call ``patch_radix_cache()`` BEFORE SGLang creates its scheduler so the
patched class is used for all cache operations.

Usage (programmatic):
    from semblend.integration.sglang.radix_patcher import patch_radix_cache
    patch_radix_cache()
    # ... then launch SGLang normally

See also: ``launch_semblend_sglang.py`` for a turnkey entry point.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("semblend.sglang.patcher")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def patch_radix_cache() -> None:
    """Monkey-patch SGLang's RadixCache with SemBlend semantic discovery.

    Replaces ``sglang.srt.mem_cache.radix_cache.RadixCache`` with a
    dynamic subclass that adds semantic donor lookup on prefix-cache miss.

    Raises:
        ImportError: If SGLang is not installed or the RadixCache module
            cannot be found.
    """
    try:
        from sglang.srt.mem_cache import radix_cache as rc_mod
    except ImportError as exc:
        raise ImportError(
            "SGLang is required for SemBlend RadixCache patching. Install with: pip install sglang"
        ) from exc

    from .radix_backend import get_semblend_radix_cache_class

    original_cls = rc_mod.RadixCache
    patched_cls = get_semblend_radix_cache_class(original_cls)
    rc_mod.RadixCache = patched_cls

    logger.info(f"Patched RadixCache: {original_cls.__name__} -> {patched_cls.__name__}")
