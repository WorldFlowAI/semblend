"""SemBlend vLLM integration — KVConnectorBase_V1 dynamic connector.

Usage with vLLM dynamic connector loading:
    --kv-transfer-config '{
        "kv_connector": "SemBlendConnectorV1",
        "kv_connector_module_path": "semblend.integration.vllm.connector_v1",
        "kv_role": "kv_both"
    }'

This module re-exports from synapse_kv_connector for backward compatibility.
The actual connector implementation lives in synapse_kv_connector/semblend_connector.py.
"""
