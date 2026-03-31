# Communication Accounting Notes

These notes summarize the questions from Problem `communication_accounting`.

## Simplified XXL model

Given:

- `d_model = D = 16384`
- `d_ff = F = 53248`
- `num_blocks = L = 126`

Each block contains two linear layers:

- `D -> F`
- `F -> D`

So the parameter count per block is:

\[
2DF
\]

and total parameters are:

\[
P = 2LDF = 219{,}848{,}638{,}464.
\]

## Definitions

### What does `num_blocks` represent?

The number of repeated transformer blocks in the model. In this simplified problem, each block is only the two FFN linear layers.

### What are master weights?

The FP32 copy of the model parameters used by the optimizer during mixed-precision training. The lower-precision BF16 copy is used in forward/backward, while updates are applied to the FP32 master copy.

### What does `B` mean in the activation formula?

Here `B` is the number of token rows processed by the FFN on one device:

\[
B = \text{batch\_size} \times \text{sequence\_length}
\]

It is not just batch size and not just sequence length.

### Where does
\[
A = 2LB(D+F)
\]
come from?

For one block, the saved FFN activations are approximately:

- one `[B, D]` tensor
- one `[B, F]` tensor

So that is `B(D+F)` elements per block. Since activations are BF16, each element is 2 bytes, giving:

\[
2B(D+F)
\]

bytes per block. Multiply by `L` blocks:

\[
A = 2LB(D+F).
\]

## Memory accounting summary

### FP32 optimizer-related state on one device

- master weights: `4P` bytes
- accumulated gradients: `4P` bytes
- Adam states: `8P` bytes

Total:

\[
16P
\]

bytes.

Numerically:

\[
16P \approx 3.52 \text{ TB} \approx 3276 \text{ GiB}.
\]

### BF16 saved activations

\[
A = 2LB(D+F)
  = 17{,}547{,}264 \cdot B \text{ bytes}.
\]

That is about `16.34 MiB * B`.

### H100 80GB comparison

The FP32 state alone is about:

- `~44` H100-80GB in decimal-GB terms
- `~40.95` H100-80GiB in binary-GiB terms

## FSDP memory expression

If master weights, optimizer state, gradients, and half the activations are sharded across `N_FSDP` devices, a standard per-device expression is:

\[
M_{\text{device}}
=
\frac{16P}{N_{\text{FSDP}}}
+
\frac{A}{2N_{\text{FSDP}}}
+
\frac{A}{2}.
\]

The first term is sharded FP32 model/optimizer memory. The activation terms reflect the statement that half the activations are sharded and half are not.

## Compute-bound threshold intuition

Using the TPU Scaling Book notation from the problem:

- `W_ici = 2 * 9 * 10^10`
- `C = 4.6 * 10^14`
- `M_X = 2`
- `M_Y = 1`
- `X = 16`
- `Y = 4`

The key threshold is on per-device token batch size:

\[
\frac{B}{N} > \frac{\alpha^2}{M_X M_Y F},
\qquad
\alpha = \frac{C}{W_{\text{ici}}}.
\]

Numerically, this gives a per-device threshold of about `61.3`, so the smallest integer per-device batch is about `62`, and with `N = XY = 64` devices the overall batch is about `3968`.

## Practical throughput ideas

If we want smaller overall batch size while staying compute-efficient, common tricks include:

- combine FSDP and TP
- use pipeline parallelism with enough microbatches
- overlap communication with compute
- shard sequence/context where useful
- use expert parallelism / MoE when available

The general goal is to reduce exposed communication per token or increase compute per communicated byte.
