"""SemBlend vLLM compatibility integration - KVConnectorBase_V1 entry point.

Usage with vLLM dynamic connector loading:
    --kv-transfer-config '{
        "kv_connector": "SemBlendConnectorV1",
        "kv_connector_module_path": "semblend.integration.vllm.connector_v1",
        "kv_role": "kv_both"
    }'

This module re-exports from semblend_kv_connector for backward compatibility.
The current connector may use LMCache for KV transfer.
"""
