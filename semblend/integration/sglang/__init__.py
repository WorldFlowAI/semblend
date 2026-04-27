"""SemBlend SGLang integration -- semantic KV cache reuse for RadixAttention.

Four integration paths:

1. FuzzyMatchProvider adapter (upstreamable, target: ibifrost/sglang
   ``feature/support_fuzzy_prefix_match_opensource``):

       from semblend.integration.sglang.provider import SemBlendProviderAdapter
       from semblend.integration.sglang.config import SemBlendProviderConfig
       adapter = SemBlendProviderAdapter(SemBlendProviderConfig(...))

   The SGLang-side thin wrapper (`SemanticEmbeddingProvider`) constructs the
   adapter and forwards `cache_on_request_finished` / `match_on_prefix_miss`.

2. HiCacheStorage backend (dynamic loading, no forking):
   python -m sglang.launch_server --model-path <model> \\
       --enable-hierarchical-cache \\
       --hicache-storage-backend dynamic \\
       --hicache-storage-backend-extra-config \\
         '{"module_path":"semblend.integration.sglang.hicache_backend",
           "class_name":"SemBlendHiCacheStorage"}'

3. RadixCache monkey-patch (recommended, no forking):
   python -m semblend.integration.sglang.launch_semblend_sglang \\
       --model-path <model> --host 0.0.0.0 --port 8000

4. RadixCache class factory (programmatic):
   from semblend.integration.sglang.radix_backend import (
       get_semblend_radix_cache_class,
   )
   SemBlendCache = get_semblend_radix_cache_class(RadixCache)
"""

from semblend.integration.sglang.config import SemBlendProviderConfig
from semblend.integration.sglang.provider import SemBlendProviderAdapter
from semblend.integration.sglang.types import (
    FuzzyMatchResult,
    FuzzyMatchSegment,
    QualitySignals,
)

__all__ = [
    "SemBlendProviderAdapter",
    "SemBlendProviderConfig",
    "FuzzyMatchResult",
    "FuzzyMatchSegment",
    "QualitySignals",
]
