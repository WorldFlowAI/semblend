# Changelog

Notable changes to the `semblend` package, newest first. Releases before
0.3.22 are described by their git tags and commit history.

## 0.3.23 - 2026-09-14

### Added

**A request-derivable tenant key on every `DonorRegistered` event.** The
event's namespace now carries `extra.tenant_key`, tenant key v1:
`semblend:tenant:v1:` followed by the first 32 hex characters of
`sha256(cache_salt)`, or the sentinel `semblend:tenant:v1:none` when the
request carries no salt. It is a function of the raw salt alone, so a
gateway that sets the salt per tenant can compute the same key and a fleet
catalog can scope donor placement to one tenant without knowing anything
engine-specific. The engine-local isolation namespace at `extra.cache_salt`
is unchanged and is still what every lookup enforces. The contract and a
shared test vector live in `docs/tenant-key-v1.md` and
`tests/tenant_key_v1_vector.json`.

Publishers: the vLLM connector passes the request's salt into
`register_donor`; the TensorRT-LLM provider binds the key on the event
namespace only; the SGLang adapter now announces its donors through
`SemBlendPipeline.publish_donor_registered` (it previously published no
event at all) and carries the sentinel until its wrapper threads the salt;
the direct `SemBlendEventEmitter.donor_registered` takes `cache_salt` and
warns once when it is omitted. The raw salt never reaches an event or a log
line.

## 0.3.22 - 2026-09-13

### Security

**Semantic donor KV was not isolated by `cache_salt`.** Reported by Gufeng
through coordinated disclosure. Affects all 17 published releases from 0.1.0
through 0.3.21. Fixed in 0.3.22. A CVE has not yet been assigned; one will
be requested.

The count is 17 because PyPI is the source of truth for what was published.
This repository carries 15 tags; two published releases were never tagged
here, so the tag list understates the affected range.

**What was wrong.** SemBlend keeps a per-process store of "donor" requests
whose KV cache may be reused by later, semantically similar requests. That
store was never bound to the request's isolation key. A donor registered by
one request was visible to every later request in the same engine process,
whatever `cache_salt` either of them carried. The engine's own exact-prefix
cache honoured the salt; the semantic layer beside it did not. The core donor
store does support a namespace filter, but it treated an absent key as "every
donor is visible", so leaving the key out silently disabled isolation instead
of failing.

**What an operator was exposed to.** On an engine process shared by several
tenants and separated by distinct `cache_salt` values, a request from one
tenant could be prefilled with KV computed from another tenant's prompt. Its
output could then be influenced by, and could reflect, content from the other
tenant's context. This crosses the tenant boundary in every direction:
between two salts, from a salted request to an unsalted donor, and from an
unsalted request to a salted donor. Setting `cache_salt` per tenant, which is
the documented way to keep vLLM's prefix cache tenant-safe, gave no
protection at the semantic layer. The similarity and reuse-ratio thresholds
are quality gates, not authorisation: a near-identical prompt from another
tenant clears them. No code execution or privilege escalation is involved;
the exposure is the confidentiality and integrity of model context and
output.

**Affected paths.** All of the following, shipped in this package:

- The vLLM connector (`SemBlendConnectorV1`, loaded through
  `semblend.integration.vllm.connector_v1`, including the deprecated
  `synapse_kv_connector` alias), on both its default pipeline path and its
  legacy donor-store path (`SEMBLEND_USE_PIPELINE=0`, or the fallback taken
  when pipeline initialisation fails). The semantic layer is on by default
  once the connector is loaded.
- The SGLang radix-cache integration (the `semblend-sglang` launcher and
  `patch_radix_cache()`).
- The SGLang HiCache storage backend's donor index, and the SGLang
  `SemanticPrefixProvider` adapter. The adapter kept a namespace but passed
  the raw key through unhashed, sent `None` for unkeyed requests, and gated
  only some of its serve branches.
- Both TensorRT-LLM providers: `SemBlendTensorRTProvider`, which backs the
  `SemBlendKvConnectorScheduler` / `SemBlendKvConnectorWorker` KV connector,
  and `SemBlendProvider`, the semantic lookup provider. The KV connector
  also registered finished requests as donors without their `cache_salt`,
  so a salted tenant's donor landed in the no-salt namespace.
- The TensorRT-LLM PyTorch backend and its model-engine hook, reached
  through the `semblend-trtllm` launcher. Neither registration nor lookup
  carried a namespace, and the abstract backend contract had no parameter
  for one.
- The core multi-donor composite alignment in `semblend_core`. Two phases
  took donor ids from secondary indexes that were not namespace-filtered, so
  a composite plan could be built over content from another namespace even
  when the caller had passed a namespace correctly.
- The fleet `DonorRegistered` event emitted to the distributed donor
  catalog. The isolation key was absent from the event's namespace, so the
  catalog saw every donor in the fleet as tenant-less and could route a
  request to a donor belonging to another tenant.

The separate `semblend-vllm-connector` package is not affected; it already
bound `cache_salt` on both registration and lookup.

**The fix.** Every donor now records the namespace it was registered under,
derived from the request's `cache_salt` (or the per-request `extra_key`
SGLang derives from it), and every lookup requires exact equality on that
namespace before any similarity scoring runs. A request with no salt only
ever matches donors that also had no salt, so single-tenant deployments keep
reusing exactly as before. A composite plan built from several donors is
dropped whole if any of them is foreign. A donor whose namespace cannot be
determined is refused rather than trusted. The salt is hashed before it
reaches a donor record, an event, or a log line. Each rejection is counted
under `cross_namespace_rejected` in the integration's existing stats so that
isolation is visible to operators. Regression tests cover the two-salt,
salted-versus-unsalted, and same-tenant cases on every affected path.

**What is still true after the fix.** Three behaviours are unchanged by
design, and none of them is a defect in an engine that passes a salt:

- On the core library, `extra_key=None` still means "single tenant, show me
  every donor". That is the documented contract for direct callers and
  tightening it would break every single-tenant deployment. It is now
  audible: a store holding namespaced donors logs one warning on the first
  unnamespaced lookup and counts them all under `unnamespaced_lookups`.
- Isolation still depends on the caller supplying a key. An engine that
  never sends a `cache_salt` puts every request in one shared namespace,
  which is correct for a single tenant and silent about it otherwise.
  Defaults are unchanged, but a process that registers donors and never sees
  a salt now warns once (after `SEMBLEND_UNSALTED_DONOR_WARN_AFTER` donors,
  default 32) and reports `salted_donors_registered` /
  `unsalted_donors_registered` in `get_stats()`.
- The SGLang integrations fail closed on build shapes they cannot key
  safely: a match-prefix key that cannot be rebuilt around the donor's
  tokens, a prefix result that cannot be capped, and a `HiCacheStorage`
  whose `get`/`exists`/`batch_exists` cannot carry an `extra_info`. These
  decline the lookup rather than serve it unnamespaced, and are counted as
  `donor_key_unavailable`, `uncapped_prefix_declined` and
  `isolation_declined`. On those builds isolation holds but same-namespace
  reuse is reduced; read zero reuse there as the gate, not a regression.

**Upgrading.** No configuration change is required. Donors registered by an
earlier version are not reused after the upgrade: a donor that 0.3.21 or
earlier registered under a salt was never actually bound to that salt, so
0.3.22 cannot tell which tenant it belongs to and will not serve it. The
first request per tenant after upgrading pays a normal cold prefill. Affected
versions emit no log line or counter that would show, after the fact,
whether cross-tenant reuse took place.

Operators who cannot upgrade immediately should run one engine process per
tenant or disable the semantic layer (`SEMBLEND_ENABLED=0`). Setting
`cache_salt` alone is not a mitigation on affected versions.

Thanks to Gufeng for the report and for handling it as a coordinated
disclosure.
