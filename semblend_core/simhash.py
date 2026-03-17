"""SimHash pre-filter for fast donor candidate rejection.

Computes 64-bit SimHash of token n-gram multiset. If hamming distance > 20
(of 64 bits), the prompts are too dissimilar for donor reuse -- skip embedding.

Latency budget: <1ms (vectorized operations on token IDs, no model).
"""
from __future__ import annotations

import numpy as np

# Lookup table for popcount (bytes 0-255)
_POPCOUNT_TABLE = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.uint8
)


def compute_simhash(token_ids: list[int], ngram_size: int = 3) -> int:
    """Compute 64-bit SimHash of token n-gram multiset.

    Uses numpy vectorized operations. ~0.5ms at 4K tokens.

    Args:
        token_ids: Sequence of token IDs.
        ngram_size: Size of token n-grams (default: 3).

    Returns:
        64-bit SimHash fingerprint.
    """
    n = len(token_ids)
    if n < ngram_size:
        return hash(tuple(token_ids)) & 0xFFFFFFFFFFFFFFFF

    # Subsample for long sequences (cap at 2000 n-grams)
    stride = max(1, (n - ngram_size) // 2000)

    # Hash all sampled n-grams
    hashes = np.array(
        [
            hash(tuple(token_ids[i : i + ngram_size]))
            for i in range(0, n - ngram_size + 1, stride)
        ],
        dtype=np.int64,
    )

    # View as bytes for bit-level analysis (8 bytes per hash)
    hash_bytes = hashes.view(np.uint8).reshape(-1, 8)

    # For each byte position, count set bits using lookup table
    # Then accumulate across all hashes per bit position
    n_hashes = len(hashes)
    threshold = n_hashes // 2

    fingerprint = 0
    for byte_idx in range(8):
        byte_col = hash_bytes[:, byte_idx]
        for bit_in_byte in range(8):
            bit_idx = byte_idx * 8 + bit_in_byte
            mask = np.uint8(1 << bit_in_byte)
            set_count = int(np.count_nonzero(byte_col & mask))
            if set_count > threshold:
                fingerprint |= 1 << bit_idx

    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Compute Hamming distance between two 64-bit SimHash fingerprints."""
    return bin(a ^ b).count("1")


def bulk_hamming_distance(
    simhashes: np.ndarray, query_simhash: int
) -> np.ndarray:
    """Vectorized hamming distance using byte-level popcount lookup.

    ~10x faster than bit-shifting loop for arrays.
    """
    xor_result = simhashes ^ np.uint64(query_simhash)
    # View as bytes and use popcount lookup
    xor_bytes = xor_result.view(np.uint8).reshape(-1, 8)
    return _POPCOUNT_TABLE[xor_bytes].sum(axis=1).astype(np.int32)


def is_plausible_donor(
    query_simhash: int,
    donor_simhash: int,
    max_hamming: int = 20,
) -> bool:
    """Check if donor is a plausible match based on SimHash distance."""
    return hamming_distance(query_simhash, donor_simhash) <= max_hamming
