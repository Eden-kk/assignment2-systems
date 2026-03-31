# FlashAttention and Triton Notes

## Triton block pointers

### What does `order` mean in `tl.make_block_ptr`?

`order` tells Triton which dimension should be treated as the faster-changing layout order inside the block. For a row-major 2D tensor `[M, N]`, `order=(1, 0)` is the natural choice because columns are contiguous inside each row.

### What does `offsets` mean?

`offsets` is the starting position of the current tile inside the full tensor. For example, if `offsets=(32, 0)` and `block_shape=(16, D)`, the pointer refers to rows `32:48` and columns `0:D`.

### Why are `offsets=(0, 0)` for `K_block_ptr` and `V_block_ptr`?

Because those pointers start at the first key/value tile and are advanced inside the loop over key tiles. In contrast, `Q_block_ptr` is fixed to the query tile owned by the current program.

### How do I get the actual query row indices in the kernel?

If `query_tile_index` is the tile number, the actual query indices are:

```python
q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
```

That gives the per-row indices used for masking.

## Triton tensor math

### How do I initialize a Triton vector with `-inf`?

Use:

```python
m_i = tl.full((Q_TILE_SIZE,), float("-inf"), tl.float32)
```

### How do I compute `S_i^(j) = Q_i @ K_j^T * scale`?

Load the query and key tiles and use:

```python
s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
```

### Is `tl.dot` matrix multiplication?

Yes. In FlashAttention it is the Triton equivalent of:

```python
q_i @ k_j.T
```

### What does `m_i_j[:, None]` mean?

It adds a new dimension so a rowwise vector can broadcast across columns. If `m_i_j` has shape `[Q_TILE_SIZE]`, then `m_i_j[:, None]` has shape `[Q_TILE_SIZE, 1]`.

## FlashAttention forward formulas

### What are the next steps after computing `s_ij`?

The usual online softmax update is:

```python
m_ij = tl.max(s_ij, axis=1)
m_new = tl.maximum(m_i, m_ij)

alpha = tl.exp(m_i - m_new)
p_tilde = tl.exp(s_ij - m_new[:, None])

l_i = alpha * l_i + tl.sum(p_tilde, axis=1)
o_i = alpha[:, None] * o_i + tl.dot(p_tilde, v_j)
m_i = m_new
```

### Where is the causal mask applied?

In both the ordinary attention implementation and FlashAttention, the mask is applied to the score matrix before softmax. For a score tile `S_i^(j)`, masking depends on the relation between query tile `i` and key tile `j`.

### How do I apply the causal mask in FlashAttention-2?

Build row and column indices:

```python
q_idx = q_start + tl.arange(0, Q_TILE_SIZE)
k_idx = k_start + tl.arange(0, K_TILE_SIZE)
mask = q_idx[:, None] >= k_idx[None, :]
```

Then modify scores before softmax, for example:

```python
s_ij = s_ij + tl.where(mask, 0.0, -1e6)
```

### Should `is_causal` be passed into the Triton kernel?

Yes. Preferably as a `tl.constexpr` parameter like `IS_CAUSAL`, so Triton can specialize the kernel.

## FlashAttention backward formulas

### What is `dO` in backward?

`dO` is just the upstream gradient passed into `backward`, usually named `grad_output`.

### Where does formula (17) come from?

It is the backward formula for softmax:

\[
dS = P \odot (dP - D)
\]

where `D` is the rowwise scalar term:

\[
D_i = \sum_j dP_{ij} P_{ij}
\]

In FlashAttention this is rewritten as:

\[
D_i = \sum_d dO_{id} O_{id}
\]

so it can be computed from saved outputs instead of the full attention matrix.

### How do dense backward formulas convert to tiled backward?

The dense pattern:

```python
p = exp(s - l[..., None])
grad_v = p.transpose(-2, -1) @ grad_output
grad_p = grad_output @ v.transpose(-2, -1)
D = (grad_output * o).sum(dim=-1)
grad_s = p * (grad_p - D[..., None])
grad_q = grad_s @ k * scale
grad_k = grad_s.transpose(-2, -1) @ q * scale
```

becomes a nested loop over query tiles `i` and key tiles `j`, reconstructing only local `S_i^(j)`, `P_i^(j)`, `dP_i^(j)`, and `dS_i^(j)`.

## Common FlashAttention implementation gotchas

### Why do `dK` and `dV` need a different kernel ownership scheme?

If the grid is over query tiles, many programs will contribute to the same `dK_j` and `dV_j`. That causes write conflicts unless you use atomics or split the backward into:

- one `dQ` kernel owned by query tiles
- one `dK/dV` kernel owned by key tiles

### Why are real tensor strides needed in the Triton launcher?

`tl.make_block_ptr` needs the actual memory layout of the tensors, not logical sizes or tile sizes. So the launcher should pass `q.stride(...)`, `k.stride(...)`, and so on.
