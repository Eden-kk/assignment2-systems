# Optimizer Sharding and Parallelism Notes

## Optimizer state sharding

### What is the relationship between `params` and `param_group`?

`params` is the overall input to the optimizer constructor. A `param_group` is one dict inside that structure, typically containing:

- `"params"`: a list of tensors
- group-specific optimizer metadata such as `"lr"`, `"weight_decay"`, `"betas"`, and `"eps"`

### Why use param groups if parameters are updated independently?

Param groups are for configuration and bookkeeping, not mathematical dependency. They let different parts of the model use different optimizer hyperparameters.

### Should parameter-group metadata be handled generically?

Yes. Do not hard-code optimizer-specific keys if you want the wrapper to support arbitrary optimizers. Treat `"params"` as the special field and preserve the rest of the key-value pairs generically.

### What are `betas` and `eps` for?

For Adam/AdamW:

- `betas = (beta1, beta2)` control the moving averages of gradients and squared gradients
- `eps` is a small numerical-stability constant added in the denominator

### Where are gradients stored?

On the parameters themselves, as `param.grad`. The optimizer state is separate and stored in `optimizer.state[param]`.

### How is `zero_grad()` used?

At the start of each training step, before the next backward pass, to clear stale gradients.

### Should params be sharded before or after `super().__init__(params, defaults)`?

The sharding logic should live in `add_param_group(...)`, because the base `Optimizer.__init__` calls `self.add_param_group(...)`. That means the sharding behavior must be ready before or during superclass initialization.

### Should the sharding logic be in `__init__` or `add_param_group`?

Primarily in `add_param_group(...)`, because `add_param_group(...)` may be called both during construction and later during training. `__init__` should mainly set up distributed state and bookkeeping.

### Does `__init__` still need code if sharding is in `add_param_group`?

Yes. It still needs to initialize:

- `self.rank`
- `self.world_size`
- `self.optimizer_cls`
- any bookkeeping containers used by `add_param_group(...)`

### Can I assume the number of params in a group is divisible by `world_size`?

No. Use a rule like modulo assignment:

```python
owner_rank = param_index % world_size
```

### How do I quickly get local params from a full list?

```python
local_params = [param for i, param in enumerate(params) if i % world_size == rank]
```

### Should `self.param_groups` stay global or become local?

A clean design is:

- `self.param_groups`: global/logical optimizer view
- `self.local_param_groups`: local shard used for actual updates

This preserves the standard optimizer interface while still sharding the heavy optimizer state.

### If full `param_groups` stay around, where do memory savings come from?

From sharding optimizer state, not from deleting lightweight metadata. Param-group dicts are tiny compared with Adam moment tensors.

### What gets updated locally in `step()` and what gets communicated?

Locally updated:

- this rank’s owned parameters
- this rank’s optimizer state for those parameters

Communicated after local update:

- updated parameter shards, so all ranks end with the same full model weights

### Is there a place for `barrier()` or sync in sharded optimizer `step()`?

Usually no extra `dist.barrier()` is needed for correctness. The parameter broadcasts themselves are collectives. Device sync is mainly for benchmarking, not correctness.

## Parallelism concepts

### Which kind of parallelism is overlap bucketed DDP?

It is still data parallelism (DP). The model is replicated, the batch is split across ranks, and the gradients are synchronized. Overlap and bucketing change the communication strategy, not the parallelism type.

### How is tensor parallelism (TP) realized in code?

By sharding tensors inside layers across ranks. For example:

- column-parallel linear layers shard the output dimension and often use `all_gather`
- row-parallel linear layers shard the input dimension and often use `all_reduce`

So unlike DP, TP changes the layer implementation itself and requires communication during forward/backward activations, not just gradient synchronization at the end.
