"""SemBlend SGLang integration -- semantic KV cache reuse for RadixAttention.

Three integration paths:

1. HiCacheStorage backend (dynamic loading, no forking):
   python -m sglang.launch_server --model-path <model> \\
       --enable-hierarchical-cache \\
       --hicache-storage-backend dynamic \\
       --hicache-storage-backend-extra-config \\
         '{"module_path":"semblend.integration.sglang.hicache_backend",
           "class_name":"SemBlendHiCacheStorage"}'

2. RadixCache monkey-patch (recommended, no forking):
   python -m semblend.integration.sglang.launch_semblend_sglang \\
       --model-path <model> --host 0.0.0.0 --port 8000

3. RadixCache class factory (programmatic):
   from semblend.integration.sglang.radix_backend import (
       get_semblend_radix_cache_class,
   )
   SemBlendCache = get_semblend_radix_cache_class(RadixCache)
"""
