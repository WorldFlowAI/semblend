"""Namespace construction for SGLang semantic KV reuse.

Neutral home for the isolation namespace every SGLang integration path
shares — the upstream-PR provider adapter, the HiCacheStorage backend and
the RadixCache monkey-patch — so the adapter that would travel upstream
does not import a private helper out of the monkey-patch module.

Mirrors ``semblend/integration/trtllm/namespace.py``: one hashed namespace
derived from the request's isolation value, bound at registration AND at
lookup, with an absent value mapped to a sentinel rather than to None.
"""

from __future__ import annotations

import hashlib
from typing import Any

# Isolation namespace for keys and requests that carry no extra_key.
#
# Donors and lookups always carry a non-empty namespace string so the
# isolation check is a plain equality compare with no "None means every
# donor is visible" escape hatch. Two requests that both lack an extra_key
# share this sentinel and MAY reuse each other's KV — single-tenant
# deployments, and SGLang releases whose match_prefix key is a bare token
# list, keep working unchanged. A request that carries an extra_key never
# matches this sentinel, so a keyed request can never consume an unkeyed
# donor, and an unkeyed request can never consume a keyed one.
NO_EXTRA_KEY_NAMESPACE = "semblend:ns:no-extra-key"

_EXTRA_KEY_NAMESPACE_PREFIX = "semblend:ns:extra-key:"


def isolation_namespace(extra_key: Any, cache_salt: Any = None) -> str:
    """Isolation namespace for the values a key or request carries.

    SGLang keys its own radix tree on ``extra_key`` — the field its API layer
    builds from ``cache_salt`` (and ``lora_id``) — so it is the per-request
    isolation boundary an operator already sets per tenant. SemBlend binds
    the same value to every donor at registration and requires exact
    equality at lookup. ``cache_salt`` is folded in as well for any build
    that keeps it separate from ``extra_key``.

    The value is hashed rather than stored verbatim: it is tenant-identifying
    and the namespace reaches logs and donor records.
    """
    parts = []
    for value in (extra_key, cache_salt):
        if value is None:
            parts.append("")
            continue
        text = value if isinstance(value, str) else str(value)
        parts.append(text.strip())
    if not any(parts):
        # An empty key carries no isolation intent — treat it as absent so
        # it lands in the sentinel namespace rather than in one of its own.
        return NO_EXTRA_KEY_NAMESPACE
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:32]
    return _EXTRA_KEY_NAMESPACE_PREFIX + digest


# The isolation fields, in the order isolation_namespace() takes them.
ISOLATION_FIELDS = ("extra_key", "cache_salt")

# The attributes an SGLang key or request uses to reference the objects
# beside it: MatchPrefixParams -> .key (the RadixKey) and .req (the Req).
_WRAPPER_FIELDS = ("key", "req")

# Depth bound for the walk below. Nothing SGLang passes nests anywhere near
# this far; the bound only stops an unfamiliar shape from walking forever.
_MAX_CHAIN_DEPTH = 8


def _linked_objects(root: Any) -> tuple[Any, ...]:
    """``root`` and everything reachable from it through the wrapper fields.

    Breadth-first from ``root`` so the outermost object wins a field it
    actually carries, identity-deduplicated so a key that references its own
    request (or vice versa) terminates, and depth-bounded.

    Containers are skipped: a bare token list is the whole key on older
    SGLang releases and carries no isolation value to find.
    """
    seen: set[int] = set()
    found: list[Any] = []
    pending: list[tuple[Any, int]] = [(root, 0)]

    while pending:
        obj, depth = pending.pop(0)
        if obj is None or isinstance(obj, (str, bytes, list, tuple, dict, set)):
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        found.append(obj)
        if depth >= _MAX_CHAIN_DEPTH:
            continue
        for field in _WRAPPER_FIELDS:
            try:
                pending.append((getattr(obj, field, None), depth + 1))
            except Exception:  # noqa: BLE001 - a property that raises is just absent
                continue

    return tuple(found)


def _field_value(obj: Any, field: str) -> Any:
    """One isolation field off one object, or None when it carries no value."""
    try:
        value = getattr(obj, field, None)
    except Exception:  # noqa: BLE001 - a property that raises is just absent
        return None
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text if text.strip() else None


def _first_field_value(chain: tuple[Any, ...], field: str) -> Any:
    """The first real value of ``field`` anywhere in ``chain``."""
    for obj in chain:
        value = _field_value(obj, field)
        if value is not None:
            return value
    return None


def chain_namespace(root: Any) -> str:
    """Isolation namespace composed from every object linked to ``root``.

    Registration is handed a request and lookup is handed a key, and which
    of the two carries which isolation field varies by SGLang build: one
    release puts the tenant's salt on the Req and the LoRA-derived extra_key
    on the RadixKey, another wraps both in MatchPrefixParams, an older one
    passes the token list alone.

    So the namespace is composed PER FIELD rather than per object: extra_key
    is taken from wherever in the chain it appears, cache_salt likewise, and
    the two are combined once. Taking the first object that carried any value
    instead would let an outer wrapper's extra_key mask an inner request's
    cache_salt, and a request whose salt sits on an object the walk never
    reached would register salted but look up unsalted — reading the pool it
    is supposed to be isolated from.

    Both sides call this with the same field set, so a request registers and
    looks up under one namespace whichever wrapper carries which field.
    """
    chain = _linked_objects(root)
    return isolation_namespace(*(_first_field_value(chain, field) for field in ISOLATION_FIELDS))
