"""Runtime state shared between the SemBlend connector and TRT-LLM model hooks."""

from __future__ import annotations

import json
import logging
import os
from threading import Lock
from typing import Any

logger = logging.getLogger("semblend.trtllm.runtime_state")

_LOCK = Lock()
_ACTIVE_PLANS: dict[int, Any] = {}


def set_active_plan(request_id: int, plan: Any) -> None:
    with _LOCK:
        _ACTIVE_PLANS[int(request_id)] = plan


def get_active_plan(request_id: int) -> Any | None:
    if os.environ.get("SEMBLEND_TRTLLM_ENGINE_BLEND", "0") != "1":
        return None
    with _LOCK:
        return _ACTIVE_PLANS.get(int(request_id))


def get_only_active_plan() -> Any | None:
    if os.environ.get("SEMBLEND_TRTLLM_ENGINE_BLEND", "0") != "1":
        return None
    with _LOCK:
        if len(_ACTIVE_PLANS) != 1:
            return None
        return next(iter(_ACTIVE_PLANS.values()))


def clear_active_plan(request_id: int) -> None:
    with _LOCK:
        _ACTIVE_PLANS.pop(int(request_id), None)


def clear_active_plans() -> None:
    with _LOCK:
        _ACTIVE_PLANS.clear()


def write_audit(payload: dict[str, Any]) -> None:
    path = os.environ.get("SEMBLEND_TRTLLM_AUDIT_PATH")
    if not path:
        return
    payload = dict(payload)
    payload.setdefault("source", "semblend.trtllm.runtime_state")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError:
        logger.debug("failed to write SemBlend TRT-LLM audit event", exc_info=True)
