# Precision Questions

This file stores short Q&A notes for precision-related questions from [cs336_spring2025_assignment2_systems.pdf](/Users/yvette/code/assignment2-systems/cs336_spring2025_assignment2_systems.pdf).

## Q: What is the difference between `fp32`, `fp16`, and `bf16`?

`fp32` uses 32 bits and gives the best numerical stability and precision, but it costs the most memory and is usually slower than lower-precision formats. `fp16` uses 16 bits, which makes it faster and smaller, but it has both lower precision and a much smaller dynamic range, so values can underflow to zero or overflow more easily during training. `bf16` also uses 16 bits, but it keeps an exponent range similar to `fp32`, so it is usually much more stable than `fp16` while still giving many of the memory and throughput benefits of lower precision.

## Q: Why does mixed precision often keep accumulations in `fp32`?

Accumulations repeatedly add small values, so rounding error compounds quickly when the accumulator itself is low precision. In `fp16`, this can produce noticeably inaccurate sums because each step is rounded with only a small mantissa and a limited dynamic range. Keeping the accumulator in `fp32` preserves much more numerical accuracy even if the inputs being accumulated were originally cast down to `fp16` or `bf16`.

## Q: What is the takeaway from `mixed_precision_accumulation`?

The `fp16` accumulator gives the worst result because repeated additions lose accuracy in low precision. Using an `fp32` accumulator is much more accurate, even when the increment value comes from `fp16`, which is why mixed-precision training usually leaves sensitive reductions and accumulations in higher precision.

## Q: For `benchmarking_mixed_precision`, how do I judge the dtype inside an autocast context?

The key idea is that `torch.autocast` chooses the dtype per operation, not per whole model, and it does not permanently change the storage dtype of model parameters. In a typical mixed-precision setup, model parameters stay in `fp32`, fast tensor-core-friendly ops such as linear layers and matrix multiplies usually run in lower precision like `fp16` or `bf16`, and numerically sensitive ops such as layer normalization, softmax-like reductions, and many losses are often kept in `fp32`.

For the toy model in the handout, a practical expectation is:

- parameters: `fp32`
- `fc1` output: usually lower precision under autocast
- `layer_norm` output: usually `fp32`
- logits after the final linear layer: usually lower precision
- loss: usually `fp32`
- gradients: usually `fp32` because the parameters are stored in `fp32`

The safest principle is to reason at the op level and then verify with `.dtype` in code instead of assuming every tensor inside the autocast block shares the same dtype.

## Q: How should I answer `benchmarking_mixed_precision` part (b)?

Layer normalization is sensitive in mixed precision because it computes means, variances, and normalized values using reductions and divisions, which are vulnerable to rounding error in low precision. In `fp16`, both the limited dynamic range and limited precision can make these computations unstable, so autocast often keeps layer norm in `fp32`. With `bf16`, the exponent range matches `fp32`, so overflow and underflow are much less of a concern, and layer norm usually does not need the same special treatment as in `fp16`, although `bf16` still has lower precision than `fp32`.
