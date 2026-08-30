# ANE-4A0 / 0b — the draft slice, and the operand substrate under it

Branch `worktree-ane4a-qwen-drafter`, from QW-3.7 (`78f71e6b`). Banked
2026-08-26.

The thesis this rung serves: **speculative decoding by changing the
physical plan of one model, rather than loading a second model.** Before
any of that can be measured, two things have to be true — the reduced
plan must be a real program, and executing it must cost what the *plan*
costs rather than what the *container* costs.

---

## ANE-4A0 — `ExecutionSlice::Draft { end }`

```
embedding -> layers [0, end) -> the component's own final norm -> head -> logits
```

A distinct variant, not an option on `LayerRange`. `LayerRange` is a
hidden-state shard that composes with other shards; a draft is a complete
model that happens to be shallower, and it owns both ends precisely so
its logits are comparable to the target's. `is_whole_stack()` is now
`Full | Draft`.

**Prefix-only by construction.** The variant cannot express `{0, 8, 16,
…}`, and that is deliberate: in a hybrid stack, omitted recurrent layers
own state transitions later layers consume, so a scattered subset is not
yet a defined program. Distributed selection is a separate rung, and the
type refuses to let it be smuggled in as a parameter.

### The gate: observational equivalence at full depth

A new traversal seam that changes the model's logits when it selects
every layer is not an experiment, it is an inference bug — and every
reduced-depth number measured through it would inherit the bug rather
than the model. So the bar is bit-identical, not a tolerance: both arms
run the same operands through the same backend in the same order, so any
difference at all is the seam's doing.

```
hybrid LLLF fixture      Draft{L} == Full   logits, final_hidden, executed_layers
                         Draft{L-1}         executes 0..L-1, finite, differs
                         Draft{0}, Draft{L+1}   refused, with the right reason

Granite 4.1 3B, 40 layers, real 6.3 GB container
                         Draft{40} == Full  100,352 logits BIT-IDENTICAL
                         Draft{39}          executes 0..39, finite, differs
```

The fixture arm runs in CI. The real-container arm is `#[ignore]`d behind
`QW38_CONTAINER` and skips loudly rather than reporting success over a
missing subject.

### Two supporting changes the gate required

- **`ExecutionTrace::executed_layers`** — the plan indices that actually
  ran. A count cannot distinguish a prefix from a planner falling back to
  the whole stack, so every assertion here is on identity, not length.
- **The recurrent-state precheck is scoped to the executed range.** It
  refuses before any output if a recurrence has nowhere to keep its
  durable buffers; unscoped, a draft was being refused — or charged
  state — for layers it never runs. `Full` is unchanged.

---

## ANE-4A0b — the operand substrate

Reading the code suggested the invariant already held. It was measured
instead, because the failure mode reading misses is exactly the
interesting one: **execution skipping layers while preparation still
reads their weights.**

Measured on the real 51 GB Qwen3.8-27B container:

```
depth  layers  operand reads  reads/layer  wall s
1      1        17            17.0          4.47
2      2        31            15.5          5.37
4      4        57            14.2          7.06
8      8       111            13.9         10.47

reads ~ 4 + 13.4 per layer      extrapolates to 863 for the full 64
peak RSS across all four arms: 26.5 GB
```

Affine in depth, with a small fixed term. **The invariant holds:**

> The cost of executing a reduced physical plan scales with the plan, not
> with the size of the authoritative container.

`ExecutionSlice` is therefore already a physical *view* over VINDEX3, not
a filter applied after the fact. **No loader change is needed.**

Why it works, for the record: `OperandStore::open` reads only segment
headers, `load_raw` seeks to one tensor and reads just its bytes, and
`PreparedOperands::load` slices the layer range *before* loading
anything. The 4-operand fixed term is the ends — embedding table, final
norm, output head.

### The economics finding this exposed

The ends do not scale with depth, and on this model they are not small:

```
embedding table   248320 x 5120  =  5.09 GB as f32
output head       248320 x 5120  =  5.09 GB as f32
```

So a drafter pays **~10 GB of resident vocabulary and ~3.6 s of the
wall time regardless of how shallow it is** — at depth 8 the ends are
still about a third of the run. A shallow drafter is cheap in layers and
expensive in vocabulary, which is a real constraint on the
accepted-tokens-per-millisecond metric and was not visible from any
projection microbenchmark.

### What is still deferred, and why

`Full` on Qwen3.8-27B extrapolates to **~93 GB of f32 resident** —
against ~80 GB effectively available, a 2 GB swap file, and a data volume
that is 100% full so swap cannot grow. That is a probable hard hang, not
an error message. The full-depth Qwen parity gate therefore stays unrun;
Granite proved the seam against real weights at a fifth of the depth and
a tenth of the footprint.

**Drafts, however, are cheap** — depth 8 on Qwen is well inside budget —
and drafts are what ANE-4A actually needs. The depth ladder can proceed
on the real target without ever materialising the full model.

---

## Next

- Real tokenisation from the container's `tokenizer.json`; `TOKENS` in
  the probe and `PROMPT_TOKENS` in the env-gated test are placeholders,
  valid but arbitrary, which is fine for a substrate measurement and not
  fine for a depth ladder.
- The depth ladder itself: KL against the target, top-1 agreement,
  target-token rank, then acceptance — on the ordinary backend, before
  the ANE enters the question at all.
- An incidental data point worth remembering when that ladder runs: on
  Granite, `Draft{39}` changed the logits but **not** the argmax
  (264 either way). One prompt, one depth, no weight at all — but it is
  the first hint that top-1 agreement may survive truncation further than
  logit-level agreement does, which is precisely the asymmetry
  speculative decoding lives on.
