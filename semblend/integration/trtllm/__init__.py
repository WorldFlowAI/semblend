"""SemBlend TensorRT-LLM integration -- semantic KV cache reuse for TRT-LLM.

Two integration paths:

1. PyTorch backend (direct KV injection, recommended):
   semblend-trtllm --engine-dir ./engines/qwen2.5-7b \\
       --host 0.0.0.0 --port 8000

2. Token substitution (C++ Executor, fallback):
   The PyTorch backend exposes KVCacheManager.get_buffers() for direct
   tensor access. The C++ Executor encapsulates KV cache -- token
   substitution redirects the radix tree to donor entries instead.

Architecture:
    TRT-LLM PyTorch ModelEngine
        -> SemBlend prefix match interception
            -> on radix tree miss: SemBlend semantic donor search
                -> compute MiniLM embedding of prompt text
                -> cosine similarity against donor store
                -> if hit: substitute donor tokens -> radix match
                -> apply RoPE correction post-load
        -> TRT-LLM loads donor's KV from paged cache (normal path)

KV Cache Layout:
    TRT-LLM PyTorch: [num_blocks, 2, tokens_per_block, num_kv_heads, head_dim]
    SemBlend standard: [num_blocks, 2, num_heads, block_size, head_dim]
    The existing Triton kernels accept explicit strides, so layout
    conversion is handled by stride computation only (zero-copy).

This module requires tensorrt_llm to be installed and is loaded at runtime.
"""
