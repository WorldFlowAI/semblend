"""SemBlend vLLM KVConnector V1 — dynamic connector entry point.

This is the module path used with vLLM's dynamic connector loading:
    --kv-connector-module-path semblend.integration.vllm.connector_v1

Requires: torch, vllm, lmcache (install with: pip install semblend[vllm])

It lazily re-exports from the canonical implementation in synapse_kv_connector
to avoid importing torch/vllm at module scope.
"""

__all__ = ["SemBlendConnectorV1"]  # noqa: F822 — lazy-loaded via __getattr__


def __getattr__(name: str):
    """Lazy import — only load torch/vllm/lmcache when vLLM requests the connector."""
    if name == "SemBlendConnectorV1":
        from synapse_kv_connector.semblend_connector import SemBlendConnectorV1
        return SemBlendConnectorV1
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
