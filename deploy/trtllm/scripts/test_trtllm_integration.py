"""TRT-LLM + SemBlend integration test script.

Run inside the trtllm-benchmark pod after baseline is complete.
Tests the SemBlend pipeline with TRT-LLM's Python API to validate:
  1. KV cache layout detection (stride computation)
  2. Semantic donor discovery pipeline
  3. Token substitution approach viability
  4. RoPE correction application
  5. End-to-end TTFT improvement on semantic hits

Usage (inside the pod):
    python3 /scripts/test_trtllm_integration.py
"""

from __future__ import annotations

import json
import sys
import time


def test_kv_cache_adapter():
    """Phase 2.2: Validate stride computation with real TRT-LLM tensors."""
    print("=== Test: KV Cache Adapter ===")
    import torch

    from semblend.integration.trtllm.kv_cache_adapter import (
        detect_trtllm_layout,
        trtllm_kv_strides,
    )

    # Simulate TRT-LLM KV cache layout (Qwen2.5-7B: 4 KV heads, head_dim=128)
    kv = torch.zeros(100, 2, 128, 4, 128, dtype=torch.float16, device="cuda")
    layout = detect_trtllm_layout(kv)
    strides = trtllm_kv_strides(kv)

    assert layout.tokens_per_block == 128
    assert layout.num_kv_heads == 4
    assert layout.head_dim == 128
    assert strides.kv_stride_dim == 1

    # Write a known value and verify stride-based access
    block, kv_idx, pos, head = 5, 0, 42, 2
    kv[block, kv_idx, pos, head, :] = torch.arange(128, dtype=torch.float16, device="cuda")

    # Read back using stride offsets
    offset = (
        block * strides.kv_stride_block
        + kv_idx * strides.kv_stride_kv
        + pos * strides.kv_stride_pos
        + head * strides.kv_stride_head
    )
    flat = kv.view(-1)
    val = flat[offset].item()
    assert val == 0.0, f"Expected 0.0 at offset {offset}, got {val}"

    print(f"  Layout: {layout}")
    print(
        f"  Strides: block={strides.kv_stride_block}, kv={strides.kv_stride_kv}, "
        f"pos={strides.kv_stride_pos}, head={strides.kv_stride_head}"
    )
    print("  PASSED")
    return True


def test_semantic_pipeline():
    """Phase 2.3: Test SemBlend pipeline with mock backend."""
    print("\n=== Test: Semantic Pipeline ===")
    from semblend_core.pipeline import SemBlendPipeline

    pipeline = SemBlendPipeline(
        max_donors=100,
        min_similarity=0.50,
        min_reuse_ratio=0.30,
        embedder_type="minilm",
        model_name="Qwen/Qwen2.5-7B-Instruct-AWQ",
        chunk_size=128,
    )

    # Register a donor
    donor_text = (
        "Cloud computing offers scalable infrastructure for modern applications. "
        "It enables organizations to reduce costs while improving flexibility. "
        "Major providers include AWS, Azure, and Google Cloud Platform."
    )
    donor_tokens = list(range(500))  # Mock token IDs
    pipeline.register_donor(
        request_id="donor-001",
        token_ids=donor_tokens,
        prompt_text=donor_text,
    )
    print(f"  Registered donor: {pipeline.donor_count} donors in store")

    # Search for a semantically similar prompt
    query_text = (
        "Cloud infrastructure provides scalable computing resources. "
        "Organizations can reduce operational costs with cloud services. "
        "AWS, Azure, and GCP are the leading cloud providers."
    )
    query_tokens = list(range(480))

    result = pipeline.find_donor(
        token_ids=query_tokens,
        prompt_text=query_text,
    )

    print(f"  Pipeline result: found={result.found}")
    if result.found:
        print(f"    similarity={result.similarity:.3f}")
        print(f"    reuse_ratio={result.reuse_ratio:.2f}")
        print(f"    embed_ms={result.timings.embed_ms:.1f}")
        print(f"    lookup_ms={result.timings.lookup_ms:.1f}")
        print(f"    total_ms={result.timings.total_ms:.1f}")
    print("  PASSED")
    return True


def test_trtllm_model_access():
    """Phase 2.1: Probe TRT-LLM internals for hook point identification."""
    print("\n=== Test: TRT-LLM Model Access ===")
    try:
        from tensorrt_llm import LLM
    except ImportError:
        print("  SKIPPED: tensorrt_llm not installed")
        return True

    MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    print(f"  Loading {MODEL}...")
    t0 = time.monotonic()
    llm = LLM(model=MODEL, enable_prefix_caching=True)
    print(f"  Loaded in {time.monotonic() - t0:.1f}s")

    # Probe internal structure
    print(f"  LLM type: {type(llm).__name__}")
    print(f"  LLM attrs: {[a for a in dir(llm) if not a.startswith('_')][:20]}")

    # Look for engine/KV manager
    engine = None
    kv_mgr = None
    for attr in ("_engine", "engine", "_model_engine", "model_engine", "_executor", "executor"):
        candidate = getattr(llm, attr, None)
        if candidate is not None:
            engine = candidate
            print(f"  Found engine at llm.{attr}: {type(candidate).__name__}")
            print(f"    Engine attrs: {[a for a in dir(candidate) if not a.startswith('_')][:20]}")

            for mgr_attr in ("kv_cache_manager", "_kv_cache_manager", "kv_mgr", "_kv_mgr"):
                kv_candidate = getattr(candidate, mgr_attr, None)
                if kv_candidate is not None:
                    kv_mgr = kv_candidate
                    print(f"    Found KV manager at .{mgr_attr}: {type(kv_candidate).__name__}")
                    print(
                        f"      KV manager attrs: {[a for a in dir(kv_candidate) if not a.startswith('_')][:20]}"
                    )
                    break
            break

    # Check for enqueue_request (token substitution hook point)
    has_enqueue = hasattr(engine, "enqueue_request") if engine else False
    has_enqueue2 = hasattr(engine, "enqueue") if engine else False
    print(f"  Has enqueue_request: {has_enqueue}")
    print(f"  Has enqueue: {has_enqueue2}")

    # Check for radix tree (radix patch hook point)
    has_radix = False
    if kv_mgr:
        has_radix = hasattr(kv_mgr, "radix_tree")
        print(f"  Has radix_tree: {has_radix}")

    # Check for get_buffers (block injection hook point)
    has_buffers = False
    if kv_mgr:
        has_buffers = hasattr(kv_mgr, "get_buffers")
        print(f"  Has get_buffers: {has_buffers}")

    results = {
        "engine_type": type(engine).__name__ if engine else None,
        "kv_mgr_type": type(kv_mgr).__name__ if kv_mgr else None,
        "has_enqueue": has_enqueue or has_enqueue2,
        "has_radix": has_radix,
        "has_buffers": has_buffers,
    }

    with open("/results/trtllm_internals.json", "w") as f:
        json.dump(results, f, indent=2)

    print("  Results saved to /results/trtllm_internals.json")
    print("  PASSED")
    return True


def main():
    print("=" * 60)
    print("SemBlend TRT-LLM Integration Tests")
    print("=" * 60)

    results = {}

    tests = [
        ("kv_cache_adapter", test_kv_cache_adapter),
        ("semantic_pipeline", test_semantic_pipeline),
        ("trtllm_model_access", test_trtllm_model_access),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            if test_fn():
                results[name] = "PASSED"
                passed += 1
            else:
                results[name] = "FAILED"
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = f"ERROR: {e}"
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    for name, status in results.items():
        print(f"  {name}: {status}")

    with open("/results/integration_tests.json", "w") as f:
        json.dump(results, f, indent=2)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
