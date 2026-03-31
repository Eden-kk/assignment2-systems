# Distributed Communication and Naive DDP Notes

## Single-node communication benchmarking

### What does `tensor_size_mb` mean?

It is the communication payload size per rank, measured in megabytes. For `float32`, the number of elements is:

```python
numel = int(tensor_size_mb * 1024 * 1024) // 4
```

### Why benchmark by memory size instead of number of elements?

Because communication cost is driven by bytes moved. Memory size is the most natural quantity for bandwidth and latency analysis, and it generalizes cleanly across dtypes.

### Should dtype also be in the benchmark config?

Yes, that is a good design choice. Even if the assignment fixes `float32`, keeping `dtype` explicit makes the benchmark clearer and easier to extend.

## Synchronization primitives

### What is the difference between `dist.barrier()`, `torch.cuda.synchronize()`, and `work.wait()`?

- `dist.barrier()`: synchronizes ranks/processes
- `torch.cuda.synchronize()`: waits for local queued GPU work to finish
- `work.wait()`: waits for a specific async collective handle

### When should `barrier()` be used in benchmarking?

Usually around phase boundaries:

- before warmup
- before timed measurement
- optionally before result gathering

Not after every iteration, because that changes what you are measuring.

### What does it mean that timings vary across ranks?

Even when every rank runs the same collective, the observed wall-clock times can differ slightly due to scheduling and communication jitter. So it is common to gather timings from all ranks and report an aggregate summary.

### On a single-node benchmark, does synchronization apply across one machine or all machines?

Across all ranks in the process group. In the current problem that means all spawned processes on the single node.

### When are `setup_process_group` and `cleanup_process_group` called?

Once per spawned worker:

- `setup_process_group(...)` at the start of the worker
- `cleanup_process_group()` at the end, usually in a `finally` block

### Should warmup and measured steps use the same function?

They usually use the same underlying communication call, but different wrappers:

- warmup: just run the collective
- measurement: run the timed helper

### How does rank 0 get timing arrays from other ranks?

A common pattern is:

```python
gathered = [None for _ in range(world_size)]
dist.all_gather_object(gathered, local_times)
```

Then rank 0 summarizes `gathered`.

## Naive DDP

### Why should non-rank-0 models be initialized from rank 0 instead of just using `config`?

`config` determines the model structure, not the parameter values. Broadcast from rank 0 ensures that every rank starts from the exact same weights.

### Should the optimizer also be broadcast?

No, not initially. Each rank can construct its own local optimizer after parameters are synchronized. If the models and gradients stay aligned, optimizer states evolve identically.

### Should `dist.barrier()` be called after broadcast?

Usually not required for correctness, because `broadcast` is already a collective. It can still be useful for debugging or for clean benchmark phase boundaries.

### How do I shard a batch for one rank?

If `batch_size` is divisible by `world_size`:

```python
local_bs = batch_size // world_size
start = rank * local_bs
end = start + local_bs
local_batch = batch[start:end]
```

### How should I compare parameters between the reference model and the DDP model?

Compare by name, not just zip order:

```python
ref = dict(reference_model.named_parameters())
ddp = dict(ddp_model.named_parameters())
```

Then compare matching names with `torch.allclose(...)`.

### Why can LM-based naive-DDP equality fail even when the algorithm is right?

Because a large LM is numerically more sensitive than a toy model. Different floating-point operation order between full-batch training and sharded-then-averaged training can create small drift, even when the algorithm is conceptually correct.

## Communication overlap

### Why use async communication if I still have to wait later?

Because the benefit is overlap. You start communication as soon as a gradient or bucket is ready, continue backward computation, and only wait later before `optimizer.step()`. That can hide some communication behind compute.

### Is overlap always more efficient than reducing the number of communication calls?

Not always. They optimize different bottlenecks:

- overlap hides communication behind compute
- fewer calls reduces collective-launch overhead

Real systems often combine both with async bucketed communication.

### What are hooks in DDP?

Hooks are callbacks registered on parameters that fire during `loss.backward()` when a parameter’s gradient becomes ready. They are the mechanism that lets DDP start communication early.

### What does `del optimizer` mean in adapter-style helper functions?

It marks the argument as intentionally unused. It does not destroy the optimizer globally; it only removes the local reference in that function.

### When are hooks called?

During backward, when autograd computes the gradient for the parameter on which the hook was registered.
