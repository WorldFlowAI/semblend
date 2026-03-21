#!/usr/bin/env bash
# Build a TRT-LLM engine for use with SemBlend.
#
# Usage:
#   bash deploy/trtllm/build_engine.sh [MODEL] [OUTPUT_DIR]
#
# Defaults:
#   MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
#   OUTPUT_DIR=./engines/qwen2.5-7b
#
# Prerequisites:
#   - NVIDIA GPU with >= 24GB VRAM (A10G, A100, H100)
#   - tensorrt_llm installed (pip install tensorrt_llm)
#   - huggingface-cli logged in (for gated models)
#
# The built engine enables:
#   - Paged KV cache with block reuse (enableBlockReuse=true)
#   - Paged context FMHA for efficient attention
#   - 128 tokens per block (SemBlend default)

set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
OUTPUT_DIR="${2:-./engines/qwen2.5-7b}"
TOKENS_PER_BLOCK="${TOKENS_PER_BLOCK:-128}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-8192}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-2048}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-10240}"
TP_SIZE="${TP_SIZE:-1}"

echo "=== SemBlend TRT-LLM Engine Build ==="
echo "Model:            ${MODEL}"
echo "Output:           ${OUTPUT_DIR}"
echo "Tokens/block:     ${TOKENS_PER_BLOCK}"
echo "Max batch size:   ${MAX_BATCH_SIZE}"
echo "Max input len:    ${MAX_INPUT_LEN}"
echo "Max output len:   ${MAX_OUTPUT_LEN}"
echo "TP size:          ${TP_SIZE}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Step 1: Convert HuggingFace checkpoint to TRT-LLM format
echo ">>> Step 1: Converting checkpoint..."
CKPT_DIR="${OUTPUT_DIR}/ckpt"

python -c "
from tensorrt_llm.models import QWenForCausalLM
import tensorrt_llm

QWenForCausalLM.convert_and_save(
    '${MODEL}',
    '${CKPT_DIR}',
    tp_size=${TP_SIZE},
)
print('Checkpoint conversion complete')
" 2>&1 || {
    echo "Checkpoint conversion failed. Trying trtllm-build directly..."
}

# Step 2: Build TRT-LLM engine
echo ">>> Step 2: Building engine..."
trtllm-build \
    --checkpoint_dir "${CKPT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --use_paged_context_fmha enable \
    --tokens_per_block "${TOKENS_PER_BLOCK}" \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --max_input_len "${MAX_INPUT_LEN}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --max_num_tokens 8192 \
    --tp_size "${TP_SIZE}" \
    --workers "${TP_SIZE}"

echo ">>> Engine built: ${OUTPUT_DIR}"

# Step 3: Write SemBlend-compatible config
cat > "${OUTPUT_DIR}/semblend_config.json" << EOF
{
    "model": "${MODEL}",
    "tokens_per_block": ${TOKENS_PER_BLOCK},
    "max_batch_size": ${MAX_BATCH_SIZE},
    "max_input_len": ${MAX_INPUT_LEN},
    "max_output_len": ${MAX_OUTPUT_LEN},
    "enable_block_reuse": true,
    "enable_prefix_caching": true,
    "tp_size": ${TP_SIZE}
}
EOF

echo ">>> SemBlend config written: ${OUTPUT_DIR}/semblend_config.json"
echo ""
echo "=== Build Complete ==="
echo ""
echo "To launch with SemBlend:"
echo "  semblend-trtllm --engine-dir ${OUTPUT_DIR} --model ${MODEL} --port 8000"
echo ""
echo "To launch without SemBlend (baseline):"
echo "  semblend-trtllm --engine-dir ${OUTPUT_DIR} --model ${MODEL} --no-semblend --port 8000"
