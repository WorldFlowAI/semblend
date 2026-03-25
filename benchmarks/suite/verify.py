"""Pre-flight verification for SemBlend benchmarks.

Checks that the inference endpoints are running the expected semblend code
before benchmarks start, preventing stale-code results.

Usage:
    python -m benchmarks.suite.verify --endpoint http://localhost:8100

Called automatically by reproduce.py before each benchmark run.
"""
from __future__ import annotations

import logging
import subprocess
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Key v0.2.0 indicators
V020_INDICATORS = {
    "no_sort": {
        "check": "parts.sort() must NOT be in pipeline.py",
        "grep_pattern": "parts.sort()",
        "expected_absent": True,
    },
    "full_doc_embedding": {
        "check": "max_chars must be >= 100000 in pipeline.py",
        "grep_pattern": "max_chars.*200.000",
        "expected_present": True,
    },
}

EXPECTED_VERSION = "0.2.0"


@dataclass(frozen=True)
class VerifyResult:
    """Result of a pre-flight verification check."""

    endpoint: str
    passed: bool
    semblend_version: str
    checks: dict[str, bool]
    errors: list[str]


def verify_endpoint_health(endpoint: str, timeout: float = 10.0) -> bool:
    """Check that the inference endpoint is responding."""
    try:
        url = f"{endpoint}/v1/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception as e:
        logger.error(f"Endpoint health check failed: {e}")
        return False


def verify_k8s_pod_code(
    pod_name: str,
    namespace: str = "autoresearch",
    pipeline_paths: tuple[str, ...] = (
        "/opt/synapse/synapse_kv_connector/pipeline.py",
        "/opt/semblend_core/pipeline.py",
        "/opt/synapse/semblend_core/pipeline.py",
    ),
) -> VerifyResult:
    """Verify that a K8s pod is running v0.2.0 semblend code.

    Checks:
    1. Sentence sorting is removed (no parts.sort())
    2. Full-document embedding is enabled (max_chars >= 100K)
    3. semblend version matches expected
    """
    errors = []
    checks = {}

    # Find which pipeline.py exists
    pipeline_path = None
    for path in pipeline_paths:
        try:
            result = subprocess.run(
                ["kubectl", "exec", "-n", namespace, pod_name, "--",
                 "test", "-f", path],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                pipeline_path = path
                break
        except Exception:
            continue

    if pipeline_path is None:
        errors.append(f"No pipeline.py found in pod {pod_name}")
        return VerifyResult(
            endpoint=pod_name,
            passed=False,
            semblend_version="unknown",
            checks={},
            errors=errors,
        )

    # Check 1: No sentence sorting (exclude comments)
    try:
        result = subprocess.run(
            ["kubectl", "exec", "-n", namespace, pod_name, "--",
             "grep", "-cP", r"^\s*parts\.sort\(\)", pipeline_path],
            capture_output=True, text=True, timeout=10,
        )
        sort_count = int(result.stdout.strip()) if result.returncode == 0 else 0
        no_sort = sort_count == 0
        checks["no_sort"] = no_sort
        if not no_sort:
            errors.append(
                f"STALE CODE: pipeline.py still has sentence sorting "
                f"({sort_count} occurrences). This is pre-v0.2.0 code."
            )
    except Exception as e:
        errors.append(f"Failed to check sort: {e}")
        checks["no_sort"] = False

    # Check 2: Full-document embedding (max_chars >= 100K)
    try:
        result = subprocess.run(
            ["kubectl", "exec", "-n", namespace, pod_name, "--",
             "grep", "-c", "200.000\\|200000", pipeline_path],
            capture_output=True, text=True, timeout=10,
        )
        full_doc_count = int(result.stdout.strip()) if result.returncode == 0 else 0
        full_doc = full_doc_count > 0
        checks["full_doc_embedding"] = full_doc
        if not full_doc:
            errors.append(
                "STALE CODE: pipeline.py uses small max_chars (pre-v0.2.0). "
                "Full-document embedding requires max_chars >= 200K."
            )
    except Exception as e:
        errors.append(f"Failed to check full-doc: {e}")
        checks["full_doc_embedding"] = False

    # Check 3: Patched LMCache (PR #2803 SemanticLookupProvider)
    try:
        result = subprocess.run(
            ["kubectl", "exec", "-n", namespace, pod_name, "--",
             "python3", "-c",
             "from lmcache.v1.lookup_client.semantic_provider import SemanticLookupProvider; print('ok')"],
            capture_output=True, text=True, timeout=10,
        )
        has_lmcache_pr = result.returncode == 0
        checks["lmcache_semantic_provider"] = has_lmcache_pr
        if not has_lmcache_pr:
            errors.append(
                "MISSING PR: LMCache SemanticLookupProvider not found. "
                "Install from WorldFlowAI/LMCache@semblend/semantic-lookup-provider (PR #2803)."
            )
    except Exception as e:
        errors.append(f"Failed to check LMCache PR: {e}")
        checks["lmcache_semantic_provider"] = False

    # Check 4: Patched LMCache post-load hook (PR #2804)
    try:
        result = subprocess.run(
            ["kubectl", "exec", "-n", namespace, pod_name, "--",
             "python3", "-c",
             "import lmcache; import inspect; "
             "src = inspect.getsource(lmcache); "
             "print('ok' if 'post_load' in src or 'on_load_complete' in src else 'missing')"],
            capture_output=True, text=True, timeout=10,
        )
        has_post_load = result.returncode == 0 and "ok" in result.stdout
        checks["lmcache_post_load_hook"] = has_post_load
        if not has_post_load:
            errors.append(
                "MISSING PR: LMCache post-load hook not found. "
                "Install from WorldFlowAI/LMCache@semblend/post-load-hook (PR #2804)."
            )
    except Exception as e:
        checks["lmcache_post_load_hook"] = False

    # Check 5: ONNX GPU embedding must be active (not CPU fallback)
    try:
        result = subprocess.run(
            ["kubectl", "exec", "-n", namespace, pod_name, "--",
             "python3", "-c",
             "from semblend_core.embedder import OnnxGPUEmbedder; "
             "e = OnnxGPUEmbedder(); "
             "print('gpu' if e.available else 'cpu_fallback')"],
            capture_output=True, text=True, timeout=30,
        )
        is_gpu = result.returncode == 0 and "gpu" in result.stdout
        checks["onnx_gpu_embed"] = is_gpu
        if not is_gpu:
            errors.append(
                "ONNX GPU embedding not active — falling back to CPU. "
                "For benchmarks, GPU segmented embedding is REQUIRED. "
                "Check ONNX runtime installation and CUDA availability."
            )
    except Exception as e:
        errors.append(f"ONNX GPU check failed: {e}")
        checks["onnx_gpu_embed"] = False

    # Check 5b: Fuzzy matching must be enabled and functional
    try:
        result = subprocess.run(
            ["kubectl", "exec", "-n", namespace, pod_name, "--",
             "python3", "-c",
             "from synapse_kv_connector.alignment import FuzzyMatchConfig, "
             "compute_fuzzy_chunk_alignment; print('ok')"],
            capture_output=True, text=True, timeout=10,
        )
        has_fuzzy = result.returncode == 0
        checks["fuzzy_matching"] = has_fuzzy
        if not has_fuzzy:
            errors.append(
                "MISSING: Fuzzy chunk matching not available in synapse_kv_connector. "
                "Overlay semblend_core/alignment.py into the connector."
            )
    except Exception as e:
        errors.append(f"Fuzzy check failed: {e}")
        checks["fuzzy_matching"] = False

    # Check 5b: Fuzzy-aware bathtub curve
    try:
        result = subprocess.run(
            ["kubectl", "exec", "-n", namespace, pod_name, "--",
             "python3", "-c",
             "import inspect; from synapse_kv_connector.bathtub import compute_layer_deviations; "
             "print('ok' if 'fuzzy_fraction' in inspect.getsource(compute_layer_deviations) else 'missing')"],
            capture_output=True, text=True, timeout=10,
        )
        has_fuzzy_bathtub = result.returncode == 0 and "ok" in result.stdout
        checks["fuzzy_bathtub"] = has_fuzzy_bathtub
        if not has_fuzzy_bathtub:
            errors.append(
                "MISSING: Bathtub curve not fuzzy-aware. "
                "Overlay semblend_core/bathtub.py into the connector."
            )
    except Exception as e:
        checks["fuzzy_bathtub"] = False

    # Check 6: GPU must be A100 (not A10G)
    try:
        result = subprocess.run(
            ["kubectl", "exec", "-n", namespace, pod_name, "--",
             "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip()
            is_a100 = "A100" in gpu_name or "H100" in gpu_name
            checks["gpu_a100"] = is_a100
            if not is_a100:
                errors.append(
                    f"WRONG GPU: Found '{gpu_name}'. Benchmarks MUST run on A100 or H100, "
                    f"not A10G. Use p4d.24xlarge nodegroup."
                )
        else:
            checks["gpu_a100"] = False
            errors.append("Cannot detect GPU type via nvidia-smi")
    except Exception as e:
        errors.append(f"GPU check failed: {e}")
        checks["gpu_a100"] = False

    # Check 6: semblend version
    version = "unknown"
    try:
        result = subprocess.run(
            ["kubectl", "exec", "-n", namespace, pod_name, "--",
             "python3", "-c",
             "from importlib.metadata import version; print(version('semblend'))"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
    except Exception:
        pass

    passed = all(checks.values()) and not errors

    return VerifyResult(
        endpoint=pod_name,
        passed=passed,
        semblend_version=version,
        checks=checks,
        errors=errors,
    )


def verify_all_pods(
    pods: list[tuple[str, str]],
    namespace: str = "autoresearch",
) -> list[VerifyResult]:
    """Verify all pods and print a summary.

    Args:
        pods: List of (pod_name, description) tuples.
        namespace: K8s namespace.

    Returns:
        List of VerifyResult, one per pod.
    """
    results = []

    print("=" * 70)
    print("SEMBLEND PRE-FLIGHT CODE VERIFICATION")
    print("=" * 70)

    for pod_name, desc in pods:
        print(f"\n[{desc}] Checking pod: {pod_name}")
        result = verify_k8s_pod_code(pod_name, namespace)
        results.append(result)

        for check_name, passed in result.checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {check_name}: {status}")

        if result.errors:
            for err in result.errors:
                print(f"  ERROR: {err}")
        else:
            print(f"  All checks passed (version: {result.semblend_version})")

    print("\n" + "=" * 70)
    all_passed = all(r.passed for r in results)
    print(f"VERDICT: {'ALL CLEAR — safe to benchmark' if all_passed else 'BLOCKED — fix errors before benchmarking'}")
    print("=" * 70)

    return results


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify SemBlend code on K8s pods")
    parser.add_argument("--pods", nargs="+", default=["vllm-semblend-v020", "sglang-semblend"])
    parser.add_argument("--namespace", default="autoresearch")
    args = parser.parse_args()

    pods = [(p, p) for p in args.pods]
    results = verify_all_pods(pods, args.namespace)

    if not all(r.passed for r in results):
        print("\nBenchmarks will NOT proceed with stale code.")
        print("Fix the errors above and retry.")
        exit(1)


if __name__ == "__main__":
    main()
