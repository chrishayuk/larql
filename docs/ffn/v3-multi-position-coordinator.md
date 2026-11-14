# Design: a multi-position coordinator over multi-row remote experts

Status: **architecture approved on 2026-11-14 with the decisions recorded
below. Steps 1 to 4 and 6 of the implementation plan are implemented and
gated on the routed fixture; step 5, the GPT-OSS 20B run, is not done, so
there is still no real-model evidence for the multi-row operation.** The
sections below are the design as approved and describe the code before
those steps. It builds on the [multi-row expert operation](v3-expert-rows.md).

One departure from the text below: the block session takes a continuation
selection and refuses a provider that cannot be rewound by asking the
provider itself at construction, where the design says "from the selected
continuation's identity". Production code may not name a built-in provider,
and rollback is a capability a provider states by implementing it. Of the
shipped providers only the row provider is accepted.

Out of scope: drafting or speculation policy, acceptance heuristics, a
multi-row expert kernel, cross-client batching, a Metal coordinator, and any
performance claim. This design is about producing several causally valid
positions through one remote expert call per layer, exactly.

## What already exists

The pieces are closer together than the multi-row work assumed.

- **A CPU multi-position step exists.** `DecodeSession::step_many(tokens)`
  runs a block of tokens through the batch traversal, layer by layer: every
  position of layer 0, then every position of layer 1. It is documented as a
  verification primitive and is pinned bit for bit against per-position
  decode. It returns only the last position's logits.
- **Causality within a block already holds on that path.** When history
  exists, attention runs one position at a time inside the layer and appends
  each position's key and value before the next position reads, so position
  i sees exactly positions before it.
- **The FFN is already grouped per layer, for dense layers only.**
  `apply_from_residual_many` hands every position of a layer to the dense FFN
  at once. Its routed arm falls back to one position at a time, with a
  comment saying grouping it is "a different rung".
- **That rung's primitive now exists.** `routed_ffn_placed_rows` routes each
  row, makes one provider call for the layer and reduces each row in its own
  selection order.

What does not exist:

- **Rollback.** No CPU continuation provider has truncate, checkpoint or
  restore. Only the Metal lowered session can rewind, and it does so by
  moving a position back over device buffers, which has no CPU equivalent.
- **Per-position logits** from a block step.
- **A coordinator session that uses any of this.** The remote-expert session
  (`RoutedExpertSession`) steps one position at a time.

## Proposal

Four changes, in dependency order. Each is independently testable against
per-position decode.

### 1. Route the routed arm of the grouped FFN through one provider call

In `apply_from_residual_many`, when the layer's FFN is routed **and** a
remote provider is bound, build one `RoutedFfnCall` per position and call
`routed_ffn_placed_rows` once. Row ids are the positions' offsets within the
block. A routed layer with no provider keeps the per-position program it has
today.

The latent and shared-expert wrappers around the routed sum (bottleneck
entry and exit, the shared branch) are applied per position around the
grouped call exactly as the single-position path applies them. Remote
placement already refuses plans with shared or latent branches, so the first
rung can refuse them here too and say so.

Consequence: one remote exchange per worker per layer for the whole block,
which is the property the multi-row operation was built for.

### 2. Return every position's logits from a block step

Add a block step that returns one logits row per position. The traversal
already computes every position's final residual; today the head runs on the
last one only. Running the head on all of them is K head projections, the
same arithmetic `step` runs.

This is the output a verifier needs: row i's argmax is the model's next
token after `block[..=i]`.

### 3. Rollback on the continuation provider

Add to `ContinuationProvider`:

```text
truncate(position)   discard every position at or after `position`
```

with a default that refuses by name. Implement it for the row provider
(`RowKvState`), where it removes the trailing rows of every layer and sets
the position back. Refuse, in the first rung:

- any provider that discards history it would need to restore (the window
  provider keeps only the required range; a rewind across what it dropped
  cannot be undone),
- the codec providers,
- recurrent state and latent rows. A recurrent layer's state after position
  i is not recoverable from its state after position i+k without a
  checkpoint. That is a separate design.

Refusal happens when the block session is created, from the selected
continuation's identity and the plan's layer kinds, not when the first
rewind is attempted.

Truncating forward, or to a position before the block's start, refuses.

**Truncate restores logical state, not a counter.** After `truncate(p)` the
provider must be indistinguishable, to every later read, from one that only
ever advanced to `p`. Moving the position back is not sufficient on its own.
The condition that makes this exact:

- A provider is rewindable to `p` only if every row that attention at any
  position from `p` onwards may read is still held. With a sliding window of
  width `w`, position `q` reads rows `q − w` to `q − 1`; after a rewind to
  `p`, the next position reads from `p − w`.
- The row provider keeps every row of every layer and never evicts, so it
  satisfies this at any `p`: truncation drops the rows at and after `p` and
  nothing a later position needs was ever discarded. The window is applied
  when attention reads, not when rows are stored.
- A provider that evicts or overwrites (the window provider keeps only the
  required range; a ring buffer would overwrite) may have discarded rows
  that a rewound position needs. Such a provider must either stage the
  block's writes so nothing is evicted until commit, or refuse. In the first
  rung every such provider refuses when the block session is created,
  before any state moves.

This is a correctness condition, and it is tested directly: blocks that
start before, straddle and start beyond the window boundary, with partial
commits at every count, each followed by decode that must match a session
which never saw the discarded positions.

### 4. A block coordinator session

A session beside `RoutedExpertSession`, over coordinator operands with a
bound provider, with three operations:

```text
verify(tokens)  -> logits per position   run the block; state advanced but uncommitted
commit(count)                              keep the first `count` positions, discard the rest
position()                                 the committed position
```

- `verify` records the committed position as a mark, runs the block step and
  returns every row's logits. The continuation now holds the block's rows.
- `commit(count)` truncates the continuation to `mark + count` and advances
  the committed position to match. `commit(0)` discards the whole block.
- A second `verify` before `commit` refuses: there is at most one uncommitted
  block.

Acceptance is the caller's decision. The session does not compare drafts with
argmax; it reports logits and keeps or discards what it is told to. That
keeps drafting policy out of the execution path, as the Metal
`verify` / `truncate` pair does.

## Correctness questions and how each is answered

**Causal attention across the block.** Unchanged from `step_many`: within a
layer, positions are attended in order and each appends before the next
reads. The grouped FFN call happens after the layer's attention for every
position, and the FFN of position i depends only on position i's residual,
so grouping it cannot leak a later position into an earlier one. The gate is
bitwise equality of every position's logits with per-position decode.

**Uncommitted state.** Between `verify` and `commit` the continuation holds
rows for positions that may be discarded. Nothing else observes them: the
session's committed position does not move, and a second `verify` is refused.

**Rollback without corrupting later inference.** After `commit(count)`, the
continuation must be indistinguishable from one that decoded exactly
`count` more positions one at a time. The gate: after any partial commit,
continue decoding and compare logits bit for bit with a session that never
saw the discarded positions, across the sliding-window boundary.

**Atomic failure of a remote batch.** A provider error arrives after this
layer's attention has appended rows for every block position, and after
earlier layers ran completely. Two options:

- *A. Invalidate, as today.* A failed step marks the session failed; the
  caller builds a fresh session and replays the committed prefix. Simple,
  already the single-row contract, and correct.
- *B. Recover in place.* Truncate back to the mark and stay usable.

The first rung should use A. It needs nothing new, and B's only extra
requirement is the same `truncate` this design adds, so it can follow once
rollback is proven. Either way no partial block is ever committed: the
multi-row call returns every pair or an error, and the block step returns
logits only if every layer succeeded.

**Deterministic reduction.** Unchanged and already tested: each row is
reduced in its own production selection order, whatever order workers answer
in.

**Provider ownership.** The session takes the admitted provider from
`admit_expert_provider` or coordinator operands that already bind one. It
does not re-derive ownership. Rows carry only an offset within the block as
their id; the worker learns nothing about positions or sessions.

**Routing trace.** A block records all rows of a layer consecutively, where
per-position decode records one layer per position in turn. The same
selections are recorded in a different order, so a capture taken through a
block is not index-comparable with a per-position capture. Any consumer that
compares captures by index must be told which it has.

## Gates for the first rung

All bitwise, on the routed fixture and on GPT-OSS 20B:

1. **Block equals per-position.** For K in {1, 2, 4, 8}, every position's
   logits from one block step equal per-position decode, with local experts
   and with remote experts over one and two workers.
2. **Across the window.** The same, with the block straddling the
   sliding-window boundary and starting beyond it.
3. **Commit all.** After `commit(K)`, further decode equals a per-position
   session.
4. **Partial commit.** For every `count` in 0..K, further decode after
   `commit(count)` equals a session that decoded only those positions.
5. **Repeated verify and discard** at the same mark gives the same logits
   each time.
5a. **Chained blocks.** Verify eight positions, commit three, verify four
   more, commit two, then decode normally. Every logits row of both blocks
   and every later position must equal a reference that processed only the
   five committed positions and never saw a rejected one. Run with the
   chain crossing the window boundary. This is the gate for stale state that
   a single verify and commit would not expose.
6. **One exchange per worker per layer** for the block, counted at the
   transport, and no single-row exchange.
7. **Failure at every layer.** A worker fault injected at the first, a
   middle and the last layer of a block each leaves the committed position
   unchanged, yields no logits, marks the session failed, and a fresh
   session replaying the committed prefix matches. No part of the failed
   block is ever committed.
8. **Refusals.** Unsupported continuation providers, recurrent or latent
   layers, a second uncommitted block, a forward truncate, and a routed
   layer with shared or latent branches are each refused by name before any
   state moves.
9. **Mutation checks.** Skipping the truncate, off-by-one in the kept count,
   and feeding position 0's residual to every row must each fail a gate.

## Decisions (approved 2026-11-14)

1. **A failed block invalidates the session.** Recovery is a fresh session
   replaying the committed prefix, as for a failed single step. In-place
   recovery is not part of this rung.
2. **A sibling block-session type.** The single-position session's contract
   and tests are untouched.
3. **The CPU row continuation provider only.** Every other provider, and any
   plan with recurrent or latent state, is refused at session creation.
4. **Logits for every verified position.**
5. **`truncate(position)` joins the continuation contract**, defaulting to
   refusal, with the logical-state semantics above.

Also fixed by the approval: use the existing `step_many` traversal and
`routed_ffn_placed_rows`; do not introduce a second block execution engine.
Scope is exact multi-position correctness only: no drafting, no acceptance
policy, no multi-row kernel, no cross-client scheduling, no throughput claim.

## Implementation plan

In order; each step lands with its gates before the next starts.

1. **`truncate` on `ContinuationProvider`** with a refusing default, and the
   row provider's implementation. Gates: rewind then re-advance equals never
   having advanced, per layer, across the window; refusals for forward
   truncation and for every other shipped provider.
2. **Per-position logits from the block traversal.** Gate: every row equals
   `step` at that position, local execution, K in {1, 2, 4, 8}.
3. **Routed arm of `apply_from_residual_many` through
   `routed_ffn_placed_rows`** when a provider is bound. Gates: block equals
   per-position with remote experts over one and two workers; exactly one
   multi-row exchange per contacted worker per layer, counted at the
   transport; no single-row exchange.
4. **The sibling block session** (`verify`, `commit`, `position`) with
   admission refusals. Gates 3 to 8 above, including chained blocks and
   failure at every layer.
5. **Real model.** GPT-OSS 20B, block equals per-position token ids and
   logits through remote experts. This is the first real-model evidence for
   the multi-row operation.
6. **Mutation checks** (gate 9) against the finished implementation.

One finding from review of the multi-row change bears on step 3: the
coordinator's multi-row dispatch does not currently forward its transport's
profile records to the calling thread, so request counts must be taken at
the transport, not from a profile, until that is fixed.

## What this rung would and would not show

It would show that K causally valid positions can cross the network once per
layer and produce exactly the logits per-position decode produces, with
state that can be kept or discarded exactly.

It would not show any speed-up. Whether accepted tokens grow faster than
network crossings depends on drafting and acceptance, which are not designed
here, and on worker cost per row, which is unchanged because the expert
kernel still takes one row at a time.
