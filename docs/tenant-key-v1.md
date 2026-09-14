# Tenant key v1

`tenant key v1` is the contract for the tenant identity a SemBlend worker
publishes on the semantic-KV event plane. It is versioned by name: any change
to the derivation ships as `v2` under a new prefix, never as a redefinition of
this one.

Reference implementation:
`semblend/integration/dynamo/semantic_events.py`
(`tenant_key_for_salt`, `bind_tenant_key`, `TENANT_KEY_EXTRA_FIELD`,
`NO_TENANT_KEY`).

## Derivation

```
tenant_key := "semblend:tenant:v1:" + sha256(cache_salt utf-8).hexdigest()[:32]
```

when the request carries a non-empty `cache_salt`, and the sentinel

```
tenant_key := "semblend:tenant:v1:none"
```

when it does not.

Two properties define it:

- **It is a function of the raw `cache_salt` alone.** Nothing else — not the
  model, the tokenizer, the block size, the dtype, the LoRA adapter, nor any
  engine-private namespace string — takes part. A caller that holds the
  request (a gateway that sets one salt per tenant, for instance) can compute
  the same string without asking the worker anything, which is the whole point
  of the field.
- **It does not reveal the salt.** The published value is a truncated SHA-256
  digest, so a tenant-identifying salt never reaches the wire or a consumer's
  logs.

The salt is hashed exactly as received — no trimming, no case folding, no
normalization — so an independent implementation reproduces the key from the
bytes it set, without having to reproduce anyone's normalization rules. An
empty string is treated as "no salt" and maps to the sentinel.

Note that the engine-local isolation key beside it is **not** derived the same
way: the vLLM connector (`semblend_kv_connector/semblend_connector.py`,
`_request_namespace`) and the TRT-LLM namespace module
(`semblend/integration/trtllm/namespace.py`, `cache_salt_namespace`) strip
surrounding whitespace from the salt before hashing it. So the tenant
partition is strictly finer than the engine-local one: two requests whose
salts differ only in surrounding whitespace share one engine-local namespace
but publish two different tenant keys. The direction is fail-safe — it
over-partitions, and can never merge two tenants — but a keyed fleet lookup
then misses donors the engine itself treats as same-tenant. A salt-setter that
wants one tenant key must send byte-identical salts.

## Where it is published

In the `DonorRegistered` event, at:

```
data.namespace.extra.tenant_key
```

beside the existing

```
data.namespace.extra.cache_salt
```

The two fields are different things and both are stamped:

| field | what it is | who can derive it |
| --- | --- | --- |
| `extra.tenant_key` | the request-derivable tenant identity defined above | anyone holding the request's salt |
| `extra.cache_salt` | the **engine-local** isolation namespace the donor's KV is actually stored under, unchanged in meaning and derivation | only the engine that registered the donor |

`extra.cache_salt` keeps its existing meaning and derivation. It stays the key
the engine enforces at lookup, and it stays useful to a consumer for
provenance and diagnostics. It is not a tenant identity a router can compute,
which is why `tenant_key` exists.

## How a consumer uses it

A fleet or router that places requests on workers filters candidate donors on
`tenant_key`:

- A **keyed** lookup sees donors carrying the identical key and nothing else.
- An **unkeyed** lookup (the caller has no key at all) sees every donor: that
  is the single-tenant fail-open contract, not an unset field to be inferred.
  A consumer that offers this escape hatch should count it.

Placement narrowing to the right tenant does not replace engine-side
enforcement. The engine still applies its own isolation namespace when it
hands back KV, so a placement can never promise reuse the engine will refuse.

## Absent field

A worker that has not been upgraded emits no `extra.tenant_key` at all. A
consumer should fall back to the engine-local `extra.cache_salt` identity in
that case (warning once), so a fleet upgraded worker by worker keeps routing.

A worker that has been upgraded but whose caller never threaded a salt
publishes the sentinel, not an absent field, so "this donor has no tenant" is
an explicit value rather than a missing one.

## Test vector

Checked in as `tests/tenant_key_v1_vector.json` (identical bytes in every
repository that implements this contract):

```json
{
  "cache_salt": "tenant-acme",
  "tenant_key": "semblend:tenant:v1:9332cc3fc0ec09d0f04fa8e1e08637b5",
  "no_salt_tenant_key": "semblend:tenant:v1:none",
  "contract": "tenant key v1"
}
```
