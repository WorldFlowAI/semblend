"""SemBlend vLLM KVConnector V1 compatibility entry point.

This is the module path used with vLLM's dynamic connector loading:
    --kv-connector-module-path semblend.integration.vllm.connector_v1

Requires for the current compatibility path: torch, vllm, lmcache.

It lazily re-exports from the canonical implementation in semblend_kv_connector
to avoid importing torch/vllm at module scope.
"""

__all__ = ["SemBlendConnectorV1"]  # noqa: F822 — lazy-loaded via __getattr__


def __getattr__(name: str):
    """Lazy import — only load torch/vllm/lmcache when vLLM requests the connector."""
    if name == "SemBlendConnectorV1":
        from semblend_kv_connector.semblend_connector import SemBlendConnectorV1

        return SemBlendConnectorV1
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
