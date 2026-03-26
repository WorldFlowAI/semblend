"""Reproducibility metadata for benchmark results.

Captures semblend version, git SHA, engine versions, hardware info,
and timestamps so every result JSON is fully traceable.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EngineVersions:
    """Versions of inference engine components."""

    vllm: str = ""
    lmcache: str = ""
    sglang: str = ""
    trtllm: str = ""


@dataclass(frozen=True)
class HardwareInfo:
    """Hardware environment for the benchmark run."""

    gpu_type: str = ""
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    instance_type: str = ""
    platform: str = field(default_factory=lambda: platform.platform())


@dataclass(frozen=True)
class ReproMetadata:
    """Complete reproducibility metadata for a benchmark run."""

    semblend_version: str
    semblend_git_sha: str
    semblend_git_dirty: bool
    timestamp_utc: str
    engine_versions: EngineVersions
    hardware: HardwareInfo
    python_version: str = field(default_factory=platform.python_version)
    run_host: str = field(default_factory=platform.node)

    def to_dict(self) -> dict:
        return asdict(self)


def _run_git(args: list[str], cwd: str | Path | None = None) -> str:
    """Run a git command and return stripped stdout, or empty string on error."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _get_semblend_version() -> str:
    """Get the installed semblend package version."""
    try:
        from importlib.metadata import version

        return version("semblend")
    except Exception:
        pass

    # Fallback: read from pyproject.toml
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text().splitlines():
            if line.strip().startswith("version"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "unknown"


def _get_git_sha(repo_path: str | Path | None = None) -> tuple[str, bool]:
    """Get git SHA and dirty status for a repo."""
    cwd = str(repo_path) if repo_path else None
    sha = _run_git(["rev-parse", "--short=12", "HEAD"], cwd=cwd)
    dirty_output = _run_git(["status", "--porcelain"], cwd=cwd)
    return sha or "unknown", bool(dirty_output)


def _detect_gpu() -> HardwareInfo:
    """Detect GPU info via nvidia-smi if available."""
    gpu_type = ""
    gpu_count = 0
    gpu_memory_gb = 0.0

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
            if lines:
                gpu_count = len(lines)
                parts = lines[0].split(",")
                gpu_type = parts[0].strip()
                if len(parts) > 1:
                    try:
                        gpu_memory_gb = float(parts[1].strip()) / 1024.0
                    except ValueError:
                        pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    instance_type = os.environ.get("INSTANCE_TYPE", "")
    if not instance_type:
        instance_type = os.environ.get("NODE_INSTANCE_TYPE", "")

    return HardwareInfo(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        gpu_memory_gb=round(gpu_memory_gb, 1),
        instance_type=instance_type,
    )


def _detect_engine_versions() -> EngineVersions:
    """Detect installed engine versions."""
    versions = {}

    for pkg, key in [
        ("vllm", "vllm"),
        ("lmcache", "lmcache"),
        ("sglang", "sglang"),
        ("tensorrt_llm", "trtllm"),
    ]:
        try:
            from importlib.metadata import version

            versions[key] = version(pkg)
        except Exception:
            versions[key] = ""

    return EngineVersions(**versions)


def collect_metadata(
    repo_path: str | Path | None = None,
    engine_versions: EngineVersions | None = None,
    hardware: HardwareInfo | None = None,
) -> ReproMetadata:
    """Collect full reproducibility metadata for a benchmark run.

    Args:
        repo_path: Path to semblend repo (for git SHA). Defaults to this repo.
        engine_versions: Override engine versions (auto-detected if None).
        hardware: Override hardware info (auto-detected if None).
    """
    if repo_path is None:
        repo_path = Path(__file__).resolve().parents[2]

    sha, dirty = _get_git_sha(repo_path)

    return ReproMetadata(
        semblend_version=_get_semblend_version(),
        semblend_git_sha=sha,
        semblend_git_dirty=dirty,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        engine_versions=engine_versions or _detect_engine_versions(),
        hardware=hardware or _detect_gpu(),
    )
