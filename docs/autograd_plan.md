# Autograd Plan (Phase A)

Status: agreed design, implementation in progress.
Style contract: full type hints, Google-style docstrings with types,
`# Import Modules` blocks, 100-char limit, mypy strict clean. The
engine stays pure stdlib - no new dependencies, permanently. Any
future accelerator is an optional, runtime-guarded fast path behind
the same op interface.

## 1. Value model

Every `ValueTensor` carries:

- `grad: Optional[list[float]]` - lazy, `None` until backward deposits;
  same length as `cells`.
- `tracked: bool` - "do ops consuming me record on the tape?"
- `learnable: bool` - "may backward deposit a `.grad` on me?"
- `version: int` - bumped by in-place mutations (`set_at`, `fill`);
  records snapshot input versions and backward raises on mismatch.

Defaults on fresh Float tensors: `tracked=True`, `learnable=True`,
`grad=None`. Non-Float tensors (Int/Bool/Str) are inert to the tape:
flags exist but recording ignores them - discrete values are constants.
Op outputs get `tracked = any(float input tracked)` and
`learnable=False`; the result of an expression is never a parameter
unless the user opts in with `set_learnable`.

## 2. Flags API

- `set_learnable(elt: Any, value: Bool)` - recursive over structs,
  arrays, tensors; in place.
- `set_trackable(elt: Any, value: Bool)` - same walk.

Reading rules: `tracked` is read at record time, `learnable` at
backward time. Flip between forward and backward changes deposits,
never the already-recorded graph.

Init-time flags without boilerplate - initializer and conversion
kwargs (mechanism exists: call kwargs; `TensorInit` already carries
the `randn` seed):

    wq: Tensor[Float32, 4, 4] = randn(1, learnable = false);
    x:  Tensor[Float32, 4]    = zeros(trackable = false);
    w:  Tensor[Float32, 2, 3] = array_to_tensor(data, learnable = false);

`detach(x)` (Phase B) - functional untracked copy of an expression
result.

## 3. Tensor <-> Array conversions

- `tensor_to_array(t)` - rank-preserving nested row-major arrays
  (rank 1 flat, rank 2 array of rows). Round-trips exactly.
- `array_to_tensor(arr, learnable = true, trackable = true)` -
  self-contained: shape derived from nesting, ragged nesting raises
  `ValueError` immediately; element type from the array's declared
  type (or inferred from the first cell). The existing assignment
  type check enforces the target shape match - a mismatch raises
  `ValueError` there. Schema-default interception follows the
  `TensorInit` pattern: keep as `default_expr`, validate at
  materialization.

## 4. Tape

One tape per top-level execution pass; owned by the `ExecutionScope`
(threaded like `globals_dict`), discarded when the pass ends.
Gradients deposited on attributes survive; the tape never persists.

A `Record` holds:

- `rule` - callable from the `BACKWARD_RULES` registry.
- `inputs` - consumed `Value`s by identity.
- `output` - produced `Value` by identity (may be a scalar `Value` -
  the loss is one).
- `saved` - plain data (shapes, argmax positions, indices, inverse
  permutations) immune to later mutation.
- `versions` - snapshot of input version counters.

Recording choke points (three, all in `expression.py`):

1. The member-call site - after a callable returns, if the member
   name is on the differentiable allowlist (`matmul`, `sum`, `mean`,
   `max`, `min`, `transpose`, `permute`, `reshape`, `flatten`,
   `concat`, `gather`, `relu`, and later `sigmoid`/`gelu`/`softmax`/
   `exp`) and any Float input is tracked.
2. `BinaryOpExpr._eval_tensor_arithmetic` - add/sub/mul/div.
3. `IndexExpr` slice reads - backward scatters into a zero tensor of
   the source shape.

`TensorCallable` itself stays autograd-free. `map` is permanently
gradient-free - differentiable code uses named ops. `set_at`/`fill`
never record; they bump the version instead.

## 5. Backward

`backward(loss)` - a builtin; reverse walk over the tape:

1. Seed the loss grad with ones.
2. Per record (reverse = topological): pull the output's accumulated
   grad; `None` means a branch that never reached the loss - skip.
   Call the rule; elementwise-accumulate per-input grads into a
   buffer keyed by value identity.
3. Deposit: every recorded input with `learnable=True` and a buffer
   entry gets `.grad` (assign or elementwise `+=` if one exists).
4. Clear the tape.

## 6. Per-op gradient rules (Phase A)

Broadcasting is only two cases (same-shape, tensor-scalar), so
"unbroadcast" is identity or a plain sum.

- add: `da = g`, `db = g`; sub: `da = g`, `db = -g`.
- mul: `da = g*b`, `db = g*a`; div: `da = g/b`, `db = -g*a/b^2`.
- matmul 2-D: `da = g @ b^T`, `db = a^T @ g`.
- dot (1-D): `da = g*b`, `db = g*a`.
- matvec: `da = outer(g, b)`, `db = column sums of a scaled by g`.
- vecmat: `da = g @ b^T`, `db = outer(a, g)`.
- sum: broadcast `g`; mean: `g / n`.
- max/min: zeros except the argmax cell (index saved at record time).
- transpose/permute: inverse permutation - free.
- reshape/flatten: reshape back (shape saved).
- concat: split `g` along the join axis.
- gather: scatter-add rows (duplicate indices accumulate - the
  embedding gradient).
- slice read: zeros of the source shape, block assigned.
- relu: `g` masked by input > 0.

Phase B: conv1d/conv2d (kernel grad = correlation of input patches
with output grad; input grad = correlation with the 180-degree
rotated kernel, spread by stride), max/avg pooling (route to winners
/ divide), sigmoid/gelu/softmax/exp members.

## 7. Activations

Named tensor members, never `map`:

- `relu` (Phase A), `sigmoid`, `gelu` (tanh approx), `exp`,
  `softmax` (1-D whole / 2-D per row, max-subtracted) - Phase B.
- Softmax backward: `da_i = y_i * (g_i - sum_j g_j * y_j)` per row.

## 8. Optimizers

- One internal `ParamWalk` utility: yields (locator, tensor) for
  every Float tensor with a grad, recursing structs/arrays; locator
  is the attribute path tuple (stable across persistence, unlike
  identity).
- Optimizers are built-in structs following the `DateTime` pattern:
  `opt = SGD(lr = 0.05);` then `opt.step(model)` / `opt.zero_grad(model)`.
  Hyperparameters are plain mutable attributes (schedules are just
  assignments in the loop).
- State dict keyed by locator; in-memory for Phase A, persistence
  later (the locator design enables it).
- Rollout: SGD (+ momentum) Phase A, Adam Phase B, AdaFactor/AdamW
  later. Each is one struct + one update rule on the shared skeleton.

## 9. Data pipeline

- `get_batch_of_partition(arr, seed, partition, batch_size,
  crt_partition, iteration) -> Array[T]`
  - Semantic (a): `iteration` counts draws for THIS partition; the
    caller keeps one counter per partition and resets to 0 per epoch.
  - One global seed-fixed permutation for the whole run (train/valid/
    test from the same shuffle - stable partition membership, no
    leakage). No re-shuffle ever.
  - Sizing: largest-remainder over `len(arr) * weight`; shared
    internal helper so both functions agree exactly.
  - Short tail batch, hard-stop empty array at exhaustion.
  - Validation: weights positive and summing to 1 within 1e-9
    (Float64 makes exact comparison false-positive), partition index
    in range, iteration >= 0, batch_size >= 1 (Int32).
- `get_partition_iteration_count(arr, seed, partition, batch_size,
  crt_partition) -> Int64` - same interface minus iteration; shared
  sizing helper.
- Both accept an empty array (count 0, empty batches).

## 10. Engine qualities (non-negotiable)

- Simple, modular, clean, beautiful, readable - understandable by
  anyone. One shared partition helper, one tape class, one rules
  registry; no cleverness.
- Fuel counters stay where they are (user loops). Tapes are bounded
  by the recorded DAG, which is bounded by fueled execution.
- In-place mutation of tracked tensors between forward and backward
  raises loudly (version check) instead of corrupting results.

## 11. Phases

- Phase A: tape + rules + hooks + backward + setters + init-kwargs +
  conversions + ParamWalk + SGD + relu + data pipeline. Acceptance:
  seeded 2-layer MLP trained inside pytest via
  `get_batch_of_partition`, final loss pinned like every golden test.
- Phase B: conv/pool rules, sigmoid/gelu/softmax/exp, Adam, detach.
- Phase C: optimizer-state persistence, schedules, trainable
  transformer flagship example.
