"""Tests that the vLLM connector can be imported via the new module path.

Note: These tests verify the import paths only. They do NOT instantiate
the connector (which requires vLLM + LMCache + CUDA). The actual E2E
connector tests run on GPU infrastructure.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_vllm_imports():
    """Mock vLLM and LMCache imports for testing on CPU."""
    mock_modules = {}
    for mod_name in [
        "vllm",
        "vllm.config",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        "vllm.forward_context",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.kv_cache_manager",
        "vllm.v1.core.sched",
        "vllm.v1.core.sched.output",
        "vllm.v1.kv_cache_interface",
        "vllm.v1.request",
        "lmcache",
        "lmcache.integration",
        "lmcache.integration.vllm",
        "lmcache.integration.vllm.lmcache_connector_v1",
        "torch",
        "torch.cuda",
        "triton",
    ]:
        if mod_name not in sys.modules:
            mock_modules[mod_name] = MagicMock()
            sys.modules[mod_name] = mock_modules[mod_name]

    # Set up KVConnectorBase_V1 mock
    mock_base = MagicMock()
    mock_base.KVConnectorBase_V1 = type("KVConnectorBase_V1", (), {})
    sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"] = mock_base

    yield

    # Cleanup
    for mod_name in mock_modules:
        if mod_name in sys.modules and sys.modules[mod_name] is mock_modules[mod_name]:
            del sys.modules[mod_name]

    # Clear cached imports
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("synapse_kv_connector") or mod_name.startswith(
            "semblend.integration.vllm"
        ):
            if mod_name in sys.modules:
                del sys.modules[mod_name]


class TestVllmImportPaths:
    """Verify both old and new import paths resolve."""

    @pytest.mark.skip(reason="Requires full vLLM mock — tested in E2E")
    def test_new_path_import(self):
        """The new semblend.integration.vllm.connector_v1 path."""
        mod = importlib.import_module("semblend.integration.vllm.connector_v1")
        assert hasattr(mod, "SemBlendConnectorV1")

    @pytest.mark.skip(reason="Requires full vLLM mock — tested in E2E")
    def test_old_path_import(self):
        """The old synapse_kv_connector.semblend_connector path."""
        mod = importlib.import_module("synapse_kv_connector.semblend_connector")
        assert hasattr(mod, "SemBlendConnectorV1")

    def test_vllm_integration_init(self):
        """The semblend.integration.vllm package is importable."""
        import semblend.integration.vllm

        assert semblend.integration.vllm is not None

    def test_integration_init(self):
        """The semblend.integration package is importable."""
        import semblend.integration

        assert semblend.integration is not None
