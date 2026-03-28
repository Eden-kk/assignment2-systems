from __future__ import annotations

import math
from einops import einsum
from sympy.simplify.fu import L
import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - local macOS/dev environments may not have Triton.
    triton = None
    tl = None
    HAS_TRITON = False


DEFAULT_Q_TILE_SIZE = 16
DEFAULT_K_TILE_SIZE = 16


def _validate_flash_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[int, int, int, int]:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Expected Q, K, and V to have shape [batch, seq, d_model].")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("Q, K, and V must have the same batch size.")
    if k.shape[1] != v.shape[1]:
        raise ValueError("K and V must have the same key/value sequence length.")
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        raise ValueError("Q, K, and V must have the same hidden size.")
    return q.shape[0], q.shape[1], k.shape[1], q.shape[2]


class FlashAttention2PyTorch(torch.autograd.Function):
    """Pure-PyTorch FlashAttention-2 skeleton for forward-pass practice."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch_size, n_queries, _n_keys, d_model = _validate_flash_inputs(q, k, v)

        q_tile_size = DEFAULT_Q_TILE_SIZE
        k_tile_size = DEFAULT_K_TILE_SIZE
        scale = 1.0 / math.sqrt(d_model)

        # Follow Algorithm 1 from the handout:
        # 1. Loop over query tiles i.
        # 2. Load Qi and initialize the running state for this tile:
        #       m_i in R^[Q_TILE_SIZE]
        #       l_i in R^[Q_TILE_SIZE]
        #       O_i in R^[Q_TILE_SIZE, D]
        # 3. Loop over key tiles j.
        # 4. Compute S_i^(j) = Qi @ K_j^T * scale.
        # 5. Update the running maximum m_i.
        # 6. Compute the renormalized exp scores for the current tile.
        # 7. Update l_i and O_i using the online-softmax recurrence.
        # 8. After the inner loop, normalize O_i and write O/L to global outputs.
        #
        # Tip: accumulate m_i, l_i, and O_i in float32 even if the inputs are lower precision.

        o = torch.empty_like(
            q,
            dtype=torch.float32
        )
        l = torch.empty(
            (batch_size, n_queries), 
            device=q.device, 
            dtype=torch.float32
        )

        for b in range(batch_size):
            for i in range(0, n_queries, q_tile_size):
                # load Q_i
                q_i = q[b, i: i+q_tile_size, :]
                # initialize m_i, l_i, o_i
                m_i_j_old = torch.full(
                    (q_tile_size,),
                    float("-inf"),
                    device=q.device,
                    dtype=torch.float32,
                )

                l_i_j_old = torch.zeros(
                    (q_tile_size,),
                    device=q.device,
                    dtype=torch.float32,
                )

                o_i_j_old = torch.zeros(
                    (q_tile_size, d_model),
                    device=q.device,
                    dtype=torch.float32,
                )
                for j in range(0, _n_keys, k_tile_size):
                    k_j = k[b, j:j+k_tile_size, :]
                    v_j = v[b, j:j+k_tile_size, :]
                    s_i_j = einsum(
                        q_i.to(torch.float32), k_j.to(torch.float32),
                        "query_length dimension, key_length dimension -> query_length key_length",
                    ) * scale
                    # mask if is_casual == True
                    if is_causal:
                        q_idx = i + torch.arange(q_tile_size, device=q.device)
                        k_idx = j + torch.arange(k_tile_size, device=q.device)
                        mask = q_idx[:, None] >= k_idx[None, :]
                        s_i_j = s_i_j + torch.where(
                            mask, 
                            torch.zeros_like(s_i_j),
                            torch.full_like(s_i_j, -1e6),
                        )
                    m_i_j = torch.maximum(
                        m_i_j_old, 
                        torch.max(s_i_j, dim=1).values
                    )
                    p_i_j = torch.exp(s_i_j - m_i_j[:, None])
                    l_i_j = torch.exp(m_i_j_old - m_i_j)*l_i_j_old + torch.sum(p_i_j, dim=1)
                    o_i_j = torch.exp(m_i_j_old - m_i_j)[:, None]*o_i_j_old + einsum(
                        p_i_j.to(torch.float32), v_j.to(torch.float32),
                        "q_length k_length, k_length dimension -> q_length dimension",
                    )

                    m_i_j_old = m_i_j
                    l_i_j_old = l_i_j
                    o_i_j_old = o_i_j

                o_i = o_i_j_old / l_i_j_old[:, None]
                l_i = m_i_j_old + torch.log(l_i_j_old)

                o[b, i:i + q_tile_size, :] = o_i
                l[b, i:i + q_tile_size] = l_i

        ctx.save_for_backward(l, q, k, v, o)
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.q_tile_size = q_tile_size
        ctx.k_tile_size = k_tile_size

        return o

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # The assignment explicitly allows backward to be unimplemented at first.
        raise NotImplementedError("Implement backward after you finish the forward-pass skeleton.")


if HAS_TRITON:

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        L_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        N_QUERIES,
        N_KEYS,
        scale,
        is_causal: tl.constexpr,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        # Load Qi and initialize the on-chip running state:
        #   m_i: [Q_TILE_SIZE] float32
        #   l_i: [Q_TILE_SIZE] float32
        #   O_i: [Q_TILE_SIZE, D] float32
        #
        q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        m_i_j_old = tl.full((Q_TILE_SIZE,), float("-inf"), tl.float32)
        l_i_j_old = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        o_i_j_old = tl.zeros((Q_TILE_SIZE, D,), dtype=tl.float32)
        # Loop over key tiles j from 0 to ceil_div(N_KEYS, K_TILE_SIZE).
        # Inside the loop:
        #   1. Load K_j and V_j
        #   2. Compute S_i^(j) = Q_i @ K_j^T * scale
        #   3. Update the running maximum m_i
        #   4. Compute the renormalized exp scores for this tile
        #   5. Update l_i and O_i via online softmax
        #   6. Advance K/V block pointers to the next tile
        #
        for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            k_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
            s_i_j = tl.dot(q_i, tl.trans(k_j)) * scale
            # mask if is_casual == True
            if is_causal:
                q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                mask = q_idx[:, None] >= k_idx[None, :]
                s_i_j = s_i_j + tl.where(mask, 0.0, -1e6)
            m_i_j = tl.maximum(m_i_j_old, tl.max(s_i_j, axis=1))        
            p_i_j = tl.exp(s_i_j - m_i_j[:, None])
            l_i_j = tl.exp(m_i_j_old - m_i_j) * l_i_j_old + tl.sum(p_i_j, axis=1)
            o_i_j = tl.exp(m_i_j_old - m_i_j)[:, None] * o_i_j_old + tl.dot(p_i_j, v_j)

            m_i_j_old = m_i_j
            l_i_j_old = l_i_j
            o_i_j_old = o_i_j

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        
        o_i = o_i_j_old / l_i_j_old[:, None]
        l_i = m_i_j_old + tl.log(l_i_j_old)

        # Normalize O_i by l_i, compute L_i = m_i + log(l_i),
        # and store O_i and L_i back to global memory.
        tl.store(O_block_ptr, o_i, boundary_check=(0, 1))
        tl.store(L_block_ptr, l_i, boundary_check=(0,))

else:

    def flash_fwd_kernel(*args, **kwargs):
        raise RuntimeError("Triton is not available in this environment.")


class FlashAttention2Triton(torch.autograd.Function):
    """Triton forward-pass skeleton that mirrors the PyTorch reference structure."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if not HAS_TRITON:
            raise RuntimeError("Triton is not installed, so the Triton skeleton cannot run.")

        batch_size, n_queries, n_keys, d_model = _validate_flash_inputs(q, k, v)

        q_tile_size = DEFAULT_Q_TILE_SIZE
        k_tile_size = DEFAULT_K_TILE_SIZE
        scale = 1.0 / math.sqrt(d_model)

        o = torch.empty_like(q, dtype=torch.float32)
        l = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)

        # Set the Triton launch grid to (num_query_tiles, batch_size).
        # Example shape:
        #   grid = (triton.cdiv(n_queries, q_tile_size), batch_size)
        grid = (triton.cdiv(n_queries, q_tile_size), batch_size)
        # Launch flash_fwd_kernel with the tensor pointers, strides, problem sizes,
        # scale, and constexpr tile sizes.
        flash_fwd_kernel[grid](
            q, k, v, o, l,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            l.stride(0), l.stride(1),
            n_queries, n_keys, scale, 
            is_causal=is_causal,
            D=d_model,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
        )
        # Save (logsumexp, q, k, v, output) for backward and return output.
        ctx.save_for_backward(l, q, k, v, o)
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.q_tile_size = q_tile_size
        ctx.k_tile_size = k_tile_size

        return o

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("Backward Triton kernel skeleton intentionally left unimplemented.")


__all__ = [
    "FlashAttention2PyTorch",
    "FlashAttention2Triton",
    "flash_fwd_kernel",
]
