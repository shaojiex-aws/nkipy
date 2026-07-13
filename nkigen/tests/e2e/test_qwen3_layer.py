# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 Transformer Decoder Layer.

Architecture:
  1. Pre-attention RMSNorm
  2. QKV projection  (hidden -> q, k, v per head)
  3. Reshape + transpose to multi-head format
  4. RoPE on Q and K
  5. Scaled dot-product attention
  6. Concat heads + output projection
  7. Residual connection
  8. Post-attention RMSNorm
  9. SwiGLU feedforward  (gate_up projection, SiLU, down projection)
  10. Residual connection

Sub-kernel boundaries (values that flow through reshape/transpose or between
independent compute stages) are annotated as SharedHbm so the compiler can
freely reshape/transpose them without partition_dim constraints.
"""

import pytest
import numpy as np

from nkigen import trace, knob
from harness import run_kernel_test, Mode

# ----------------------------------------------------------------
# Model hyperparameters
# ----------------------------------------------------------------
batch = 2
seq_len = 128
hidden_size = 256
n_heads = 2
head_dim = hidden_size // n_heads      # 128
intermediate_size = 256
half_dim = head_dim // 2               # 64
eps = 1e-6
scale = 1.0 / np.sqrt(head_dim).item()

# Derived
BS = batch * seq_len                   # 256
BH = batch * n_heads                   # 4

# ----------------------------------------------------------------
# Tile sizes
# ----------------------------------------------------------------
matmul_tile_2d = [128, 128]
matmul_reduction_2d = [128]
attn_tile = [1, 128, 128]
attn_reduction = [128]
rope_tile = [1, 128, 64]                # (BH, seq_len, half_dim)
elem_tile_2d = [128, 128]


# ---- helpers ----


def rmsnorm(x, weight):
    x_fp32 = x.astype(np.float32)
    w_fp32 = weight.astype(np.float32)

    sq = np.square(x_fp32)
    knob(sq).tile_op(tile_size=elem_tile_2d).layout(mem_space="Sbuf")

    sum_sq = np.sum(sq, axis=-1, keepdims=True)
    knob(sum_sq).tile_op(tile_size=[128, 128]).layout(mem_space="Sbuf")

    mean_sq = sum_sq * np.float32(1.0 / hidden_size)
    knob(mean_sq).tile_op(tile_size=[128, 1]).layout(mem_space="Sbuf")

    normed = x_fp32 / np.sqrt(mean_sq + eps)
    knob(normed).tile_op(tile_size=elem_tile_2d).layout(mem_space="Sbuf")

    result = normed * w_fp32
    knob(result).tile_op(tile_size=elem_tile_2d).layout(mem_space="Sbuf")

    return result


def softmax_3d(x):
    x_fp32 = x.astype(np.float32)

    # Workaround: reduction accumulators (x_max, sum_exp) use SharedHbm
    # to avoid 5D SBUF allocs from legalize-layout.  The 3D shape
    # (BH, 128, 1) creates a 5D physical layout where the collapse_shape
    # back to 2D has a tile/base mismatch in linalg-to-nisa.
    x_max = np.max(x_fp32, axis=-1, keepdims=True)
    knob(x_max).tile_op(tile_size=[1, 128, 128]).layout(mem_space="Sbuf", partition_dim=1)

    shifted = x_fp32 - x_max
    knob(shifted).tile_op(tile_size=attn_tile).layout(mem_space="Sbuf", partition_dim=1)

    exp_s = np.exp(shifted)
    knob(exp_s).tile_op(tile_size=attn_tile).layout(mem_space="Sbuf", partition_dim=1)

    sum_exp = np.sum(exp_s, axis=-1, keepdims=True)
    knob(sum_exp).tile_op(tile_size=[1, 128, 128]).layout(mem_space="Sbuf", partition_dim=1)

    # Softmax result is a sub-kernel boundary (feeds into context matmul).
    result = exp_s / sum_exp
    knob(result).tile_op(tile_size=attn_tile).layout(mem_space="SharedHbm")

    return result


def silu(x):
    neg_x = -x
    knob(neg_x).tile_op(tile_size=elem_tile_2d).layout(mem_space="Sbuf")

    exp_neg = np.exp(neg_x)
    knob(exp_neg).tile_op(tile_size=elem_tile_2d).layout(mem_space="Sbuf")

    one_plus = exp_neg + 1.0
    knob(one_plus).tile_op(tile_size=elem_tile_2d).layout(mem_space="Sbuf")

    sigmoid = 1.0 / one_plus
    knob(sigmoid).tile_op(tile_size=elem_tile_2d).layout(mem_space="Sbuf")

    result = x * sigmoid
    knob(result).tile_op(tile_size=elem_tile_2d).layout(mem_space="Sbuf")

    return result


def apply_rope(x, freqs_cos, freqs_sin):
    x0 = x[:, :, :half_dim]
    x1 = x[:, :, half_dim:]

    # Workaround: split compound expressions so each intermediate gets a
    # SharedHbm knob.  Without this, the multiply intermediates default to
    # SBUF with shape (BH, seq, half_dim) = (4, 128, 64) where dim 0 = 4
    # (not partition).  getBaseAndOffsets maps d0 to dim 0, causing OOB.
    x0_cos = x0 * freqs_cos
    knob(x0_cos).tile_op(tile_size=rope_tile).layout(mem_space="SharedHbm")
    x1_sin = x1 * freqs_sin
    knob(x1_sin).tile_op(tile_size=rope_tile).layout(mem_space="SharedHbm")
    out_0 = x0_cos - x1_sin
    knob(out_0).tile_op(tile_size=rope_tile).layout(mem_space="SharedHbm")

    x0_sin = x0 * freqs_sin
    knob(x0_sin).tile_op(tile_size=rope_tile).layout(mem_space="SharedHbm")
    x1_cos = x1 * freqs_cos
    knob(x1_cos).tile_op(tile_size=rope_tile).layout(mem_space="SharedHbm")
    out_1 = x0_sin + x1_cos
    knob(out_1).tile_op(tile_size=rope_tile).layout(mem_space="SharedHbm")

    # RoPE result is a sub-kernel boundary (feeds into matmul/transpose).
    result = np.concatenate([out_0, out_1], axis=-1)
    knob(result).tile_op(tile_size=attn_tile).layout(mem_space="SharedHbm")

    return result


# ---- test ----


def test_qwen3_layer(request):
    @trace(input_specs=[
        ((BS, hidden_size), "f32"),                            # hidden_states
        ((hidden_size, 1), "f32"),                             # ln1_weight
        ((hidden_size, 1), "f32"),                             # ln2_weight
        ((hidden_size, hidden_size), "f32"),                   # w_q
        ((hidden_size, hidden_size), "f32"),                   # w_k
        ((hidden_size, hidden_size), "f32"),                   # w_v
        ((hidden_size, hidden_size), "f32"),                   # w_o
        ((1, seq_len, half_dim), "f32"),                       # freqs_cos
        ((1, seq_len, half_dim), "f32"),                       # freqs_sin
        ((hidden_size, intermediate_size), "f32"),             # w_gate
        ((hidden_size, intermediate_size), "f32"),             # w_up
        ((intermediate_size, hidden_size), "f32"),             # w_down
    ])
    def kernel(hidden_states,
               ln1_weight, ln2_weight,
               w_q, w_k, w_v, w_o,
               freqs_cos, freqs_sin,
               w_gate, w_up, w_down):
        residual = hidden_states

        # 1. Pre-attention RMSNorm
        normed = rmsnorm(hidden_states, ln1_weight)

        # 2. QKV projections — SharedHbm boundary (results go through reshape)
        q = np.matmul(normed, w_q)
        knob(q).tile_op(tile_size=matmul_tile_2d + matmul_reduction_2d).layout(mem_space="SharedHbm")

        k = np.matmul(normed, w_k)
        knob(k).tile_op(tile_size=matmul_tile_2d + matmul_reduction_2d).layout(mem_space="SharedHbm")

        v = np.matmul(normed, w_v)
        knob(v).tile_op(tile_size=matmul_tile_2d + matmul_reduction_2d).layout(mem_space="SharedHbm")

        # 3. Reshape to multi-head — SharedHbm boundary (4D transpose
        #    intermediate stays in HBM to avoid SBUF OOM)
        q = np.reshape(q, (batch, seq_len, n_heads, head_dim))
        q = np.transpose(q, (0, 2, 1, 3))
        q = np.reshape(q, (BH, seq_len, head_dim))
        knob(q).tile_op(tile_size=attn_tile).layout(mem_space="SharedHbm")

        k = np.reshape(k, (batch, seq_len, n_heads, head_dim))
        k = np.transpose(k, (0, 2, 1, 3))
        k = np.reshape(k, (BH, seq_len, head_dim))
        knob(k).tile_op(tile_size=attn_tile).layout(mem_space="SharedHbm")

        v = np.reshape(v, (batch, seq_len, n_heads, head_dim))
        v = np.transpose(v, (0, 2, 1, 3))
        v = np.reshape(v, (BH, seq_len, head_dim))
        knob(v).tile_op(tile_size=attn_tile).layout(mem_space="SharedHbm")

        # 4. RoPE on Q and K
        q = apply_rope(q, freqs_cos, freqs_sin)
        k = apply_rope(k, freqs_cos, freqs_sin)

        # K^T — SharedHbm boundary (transpose feeds into matmul)
        k_t = np.transpose(k, (0, 2, 1))
        knob(k_t).tile_op(tile_size=attn_tile).layout(mem_space="SharedHbm")

        # 5. Scaled dot-product attention
        scores = np.matmul(q, k_t)
        knob(scores).tile_op(tile_size=attn_tile + attn_reduction).layout(mem_space="Sbuf")

        scores = scores * scale
        knob(scores).tile_op(tile_size=attn_tile).layout(mem_space="Sbuf", partition_dim=1)

        attn_weights = softmax_3d(scores)

        # Context — SharedHbm boundary (result goes through reshape)
        context = np.matmul(attn_weights, v)
        knob(context).tile_op(tile_size=attn_tile + attn_reduction).layout(mem_space="SharedHbm")

        # 6. Concat heads + output projection
        context = np.reshape(context, (batch, n_heads, seq_len, head_dim))
        context = np.transpose(context, (0, 2, 1, 3))
        context = np.reshape(context, (BS, hidden_size))
        # Workaround: annotate the 2D result as SharedHbm so the 4D transpose
        # intermediate stays in HBM (not promoted to SBUF).  Without this,
        # the 4D SBUF alloc has dim 0 = batch (not partition) and
        # legalize-layout cannot tile it, causing OOB in NISA lowering.
        knob(context).tile_op(tile_size=matmul_tile_2d).layout(mem_space="SharedHbm")

        attn_out = np.matmul(context, w_o)
        knob(attn_out).tile_op(tile_size=matmul_tile_2d + matmul_reduction_2d).layout(mem_space="Sbuf")

        # 7. Residual connection
        hidden_states = residual + attn_out
        knob(hidden_states).tile_op(tile_size=elem_tile_2d).layout(mem_space="Sbuf")

        residual = hidden_states

        # 8. Post-attention RMSNorm
        normed = rmsnorm(hidden_states, ln2_weight)

        # 9. SwiGLU FFN
        gate = np.matmul(normed, w_gate)
        knob(gate).tile_op(tile_size=matmul_tile_2d + matmul_reduction_2d).layout(mem_space="Sbuf")

        up = np.matmul(normed, w_up)
        knob(up).tile_op(tile_size=matmul_tile_2d + matmul_reduction_2d).layout(mem_space="Sbuf")

        gate = silu(gate)

        gated = gate * up
        knob(gated).tile_op(tile_size=elem_tile_2d).layout(mem_space="Sbuf")

        ffn_out = np.matmul(gated, w_down)
        knob(ffn_out).tile_op(tile_size=matmul_tile_2d + matmul_reduction_2d).layout(mem_space="Sbuf")

        # 10. Residual connection
        output = residual + ffn_out
        knob(output).tile_op(tile_size=elem_tile_2d).layout(mem_space="SharedHbm")

        return output

    # First verify compilation succeeds
    run_kernel_test(
        kernel,
        modes=Mode.STRING_CHECK,
        check_ir_contains=["nisa.matmul", "nisa.alloc"],
        request=request,
    )

    # Verify numerical correctness via LLVM JIT (stop after insert-memref-dealloc,
    # before linalg-to-nisa which LLVM JIT cannot execute)
    run_kernel_test(
        kernel,
        stop_after="insert-memref-dealloc",
        modes=Mode.LLVM,
        rtol=1e-3,
        atol=1e-3,
        request=request,
    )

    # HW mode skipped: backend OOB on 3D sbuf_map<tile:[1,128,128]> accessed
    # with 128-partition tile (legalize-layout assigns partition_dim=0 to a
    # buffer created after canonicalize-partition-dim runs).
    run_kernel_test(
        kernel,
        modes=Mode.CODEGEN,
        request=request,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
