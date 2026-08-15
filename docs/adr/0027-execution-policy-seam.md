# ADR-0027 — Execution-Policy Seam (`ExecutionStrategy` at the expert-group dispatch)

**Status:** Accepted, landed with BW-C's move from research instrument to
production control surface. Default is a no-op; nothing changes unless a
policy is explicitly installed.
**Affects:** new `larql-compute/src/exec_policy/`; new
`larql-compute/src/movement_ledger/decisions.rs`;
`larql-compute-metal`'s two production MoE dispatch arms
(`moe_gpu_route::encode` — the `LARQL_GPU_ROUTE=1` serve path — and
`moe_zero_copy`), plus the token boundary in `decode/mod.rs`.
**Related:** `docs/diagnoses/bwc-expert-skip-oracle.md` (BW-C1..C5, the
evidence this seam exists to act on), `docs/diagnoses/bw10-live-gate.md`
(the BW10 ledger the decisions are recorded against),
`docs/diagnoses/bwb-compact-dense-oracle.md` (BW-B, the reserved third
strategy).

---

## Context

BW-C established, on real `gpt-oss-20b` decode, that a routed MoE expert
group at a late layer is frequently deletable without changing the
generated tokens:

- **BW-C3** — 66.7% of late-layer checkpoints had the WHOLE top-4 group
  jointly removable at a 6-token horizon (attractor-controlled: 0/72
  checkpoints flagged).
- **BW-C4.5** — corrected for completion predictability, ~two-thirds of
  genuinely contested cases survive to 64 tokens. Real and substantial;
  explicitly *not* a plateau.
- **BW-C5** — an oracle repeated policy at ONE late layer skipped 227 of
  256 opportunities (88.7%), and 6 of 8 prompts kept full 32-token exact
  token parity while doing so.

All of that ran through `cpu::ops::moe::expert_override`, a one-shot
per-`(layer, expert)` ablation hook on the CPU intervention path. It can
answer "would deleting this have been safe?" It cannot delete anything on
the path that actually serves tokens, and it has no way to tell the BW10
ledger what it removed.

Two things were entangled and needed separating:

```text
router  →  "these experts conceptually participate"     (semantics)
   ?    →  "and therefore these kernels run"            (physics)
```

Until now the only way to not-run an expert was to make the router not
pick it — a different perturbation with different numerics (GPT-OSS
renormalises selected top-k weights *before* the combine, so removing a
selected expert post-hoc leaves survivors summing to `1 - w_e`, not to
1). There was no place to say "the semantic operation stands; satisfy it
differently."

## Decision

Introduce an explicit seam between the two, sited immediately before the
expensive expert kernels on the production dispatch path.

```rust
pub enum ExecutionStrategy { Canonical, Skip }

pub trait ExecutionPolicy: Send + Sync {
    fn name(&self) -> &str;
    fn expert_group(&self, site: &ExpertGroupSite) -> ExecutionStrategy {
        ExecutionStrategy::Canonical
    }
}

pub struct ExpertGroupSite {
    pub layer: usize,
    pub phase: Option<Phase>,   // from the ledger's PhaseScope
    pub step:  Option<u64>,     // token index WITHIN that phase
    pub slots: usize,
}
```

Five decisions inside that, each load-bearing:

### 1. Default `Canonical`, gated by one relaxed atomic load

With no policy installed, `decide_expert_group` returns `Canonical` after
a single `AtomicBool` load and never touches the policy lock. Production
semantics are bit-identical to an engine built without this module. Every
dispatch site is written so the canonical branch is the one that would
have existed anyway.

### 2. The unit is the GROUP, not the expert

On the GPU route the selected expert IDs are produced by
`encode_moe_router_select` and consumed by the grouped expert kernels —
the host never reads that buffer. A per-expert seam would have to read it
back, reinstating exactly the host round-trip S2 removed for a 1.40x
win. The whole-group unit is what the production path can address for
free, and it is also the unit BW-C3/C4/C5 measured.

### 3. `Skip` is encoded by the CANONICAL combine kernel at `k = 0`

`moe_weighted_combine` seeds `acc = h_post_attn[tid]` and then loops
`j < k`. At `k = 0` it writes `new_h = h_post_attn` and reads neither
`expert_outs` nor the staged down-bias — both of which a skipped group
leaves unwritten. So the residual-identity substitute comes from the same
kernel the canonical arm ends with, rather than a second identity path
that could drift from it. Everything above it — descriptor gather, both
gate/up matvecs, activation, down matvec, bias staging — is simply not
encoded.

The router still runs. That is the point of the seam: the router's answer
is unchanged, only its physical realisation is.

### 4. Decisions are recorded in the ledger, in their own counters

`movement_ledger::decisions` holds `requested / executed / skipped /
semantic_avoided / physical_avoided`. Avoided bytes are **never** folded
into `physical_touched`: a byte that never crossed the memory bus must
not appear in a bandwidth measurement. Instead

```text
physical_touched + physical_avoided  ==  what the canonical arm would have moved
```

on the covered surface, because both arms price the operation with the
same shape arithmetic (`ExpertLayerShape::movement`). That is what lets a
skipping run be compared against a canonical one.

`requested` is bumped unconditionally, including with no policy
installed. Without the denominator always running, "0% skip rate" and
"the seam was never reached" would be the same reading.

`exec_policy::resolve_expert_group` decides AND records in one call — the
same argument `coverage::record` makes for pairing the byte counters with
the coverage evidence. Two counters a caller can update independently
will eventually disagree, and a disagreement between "what the policy
did" and "what the ledger saw" is indistinguishable from a real byte
delta.

### 5. Tokens are addressed per PHASE, never globally

`exec_policy::step` keeps separate decode and prefill indices, advanced
at the same boundary that opens the ledger's `TokenScope`. A single
global counter would repeat the defect the BW-A live gate caught:
`gpt-oss-20b`'s chat template contributes ~130 prefill positions through
the same entry point real decode steps use, so "skip on token 7" against
a global counter fires inside the system prompt. An unattributed boundary
advances nothing, and `current()` returns `None` rather than a guessed
`0` — `0` being a legitimate index a policy selects on.

## What ships, and what deliberately does not

Shipped policies are deliberately stupid, and stay that way for now:

- `LayerStepMask` — named layers × a `StepSelector` (`Every`,
  `Exactly(n)`, `EveryNth(n)`, `OneOf`), optionally phase-restricted.
- `TraceReplay` — replay BW-C5's recorded `(layer, step)` oracle
  decisions on the serve path.

No predictor ships, because BW-C1 and BW-C2 already falsified the two
obvious candidates on 576 real interventions — router weight
(`point-biserial = -0.078`) and contribution-over-residual norm
(`pearson = 0.012`, `spearman = 0.038`). Layer depth is the only real
signal found (`r ≈ 0.155`), and "which layers" is what `LayerStepMask`
takes as an argument rather than guesses.

`cpu::ops::moe::expert_override` is **unchanged**. It remains BW-C's
research instrument, with a different unit and lifetime, so every
BW-C1..C5 result stays reproducible:

| | `expert_override` | `exec_policy` |
|---|---|---|
| unit | one `(layer, expert)` invocation | the whole routed group at a layer |
| lifetime | one-shot, disarms itself | standing policy |
| path | CPU `add_expert` | Metal expert dispatch |
| purpose | measure whether a deletion is safe | decide, in production, to delete |

The CPU MoE path is therefore **not** wired to this seam. Its per-expert
dispatch (three branches, one of them a spin-pool row-parallel arm) does
not have a group-level decision point without restructuring, and the
serve path is Metal.

## The gate

`crates/larql-compute-metal/tests/test_exec_policy_expert_skip.rs`, five
proofs, no model and no generation:

1. **Descriptor arm** (`LARQL_GPU_ROUTE=1` serve path) — forced skip →
   output is bitwise `h_post_attn`; ledger records exactly one avoided
   group whose byte count equals what the canonical arm measurably moved;
   zero expert bytes streamed; the `moe-experts` coverage surface does
   not fire; uninstalling restores canonical output bit for bit.
2. **Zero-copy arm** — same four claims. A seam honoured on one of two
   production arms is a seam with a hole in it.
3. **Negative control** — a mask armed on a layer this dispatch never
   reports changes nothing, bit for bit and byte for byte. Without it,
   proof 1 is equally consistent with "installing any policy breaks the
   layer".
4. **Layer addressing** — four chained layers, layer 2 masked: exactly
   one group skipped, three executed, `touched + avoided` reconstructs
   the canonical total, and the chain's output changes.
5. **Token addressing** — the same layer skips on declared decode step 0
   and runs on step 1.

## Arming it: `LARQL_EXEC_POLICY`

The seam is useless if only a test can reach it, so there is exactly one
external entry point, read by `larql run` and `larql bench`:

```text
LARQL_EXEC_POLICY=skip-layers:<L>[,<L>...]            every decode token
LARQL_EXEC_POLICY=skip-layers:<L>[,<L>...]:every-<N>  every Nth decode token
LARQL_EXEC_POLICY=skip-layers:<L>[,<L>...]:token-<N>  exactly decode token N
LARQL_EXEC_POLICY=trace:<path>                        replay a recorded trace
```

Three properties of that parser are deliberate:

- **Every form is decode-only** and cannot be made otherwise from the
  environment. Skipping during prefill perturbs the prompt's own
  representation — a different experiment with a different control, not
  something a typo in a bench command should be able to start.
- **A malformed value is a hard error**, and the CLI exits on it. The
  failure mode of a silent fallback is an A/B that compares canonical
  against canonical and reports "no change" — an instrument that cannot
  fail on known-different input.
- **An empty trace is refused** for the same reason: it would install,
  announce itself, and behave exactly like canonical execution.

Installation prints one unmissable stderr line naming the policy, so a
policy-on run can never be quoted as a baseline from scrollback.

### The steady-state A/B

Two arms, one command each, on the banked bench protocol (warmup 16,
n 256, long prompt):

```sh
# A — canonical
LARQL_GPU_ROUTE=1 LARQL_MOVEMENT_LEDGER=1 LARQL_MOVEMENT_REGIME=resident \
  larql bench --model gpt-oss-20b-q4k --warmup 16 -n 256 --prompt-file long.txt

# B — policy on (same everything else)
LARQL_EXEC_POLICY=skip-layers:20 \
LARQL_GPU_ROUTE=1 LARQL_MOVEMENT_LEDGER=1 LARQL_MOVEMENT_REGIME=resident \
  larql bench --model gpt-oss-20b-q4k --warmup 16 -n 256 --prompt-file long.txt
```

Arm B as written is the *unconditional* late-layer policy, not BW-C5's
oracle-gated one. That is the right first measurement precisely because
it separates the two questions: it deletes 100% of that layer's expert
traffic rather than the oracle's 88.7%, so it is an upper bound on the
byte saving and a lower bound on fidelity, and it answers "does removing
expert bytes move tok/s at all?" without entangling that in "was it safe
to remove them?".

The oracle-gated arm needs a trace, which
`bwc5_oracle_repeated_policy --emit-trace <dir>` now writes — one file
per prompt, since a trace addresses `(layer, decode step)` inside ONE
generation. Replay it against the SAME prompt with
`LARQL_EXEC_POLICY=trace:<file>`; against any other prompt it addresses
a trajectory that does not exist. Each file's provenance header carries
the layer, lookahead, generation length, prompt text, skip count and
whether that prompt held full parity, so a replay result cannot be
quoted without what produced it.

Decode step 0 is the FIRST GENERATED token in both places — prefill
positions carry their own phase index and are never skipped — which is
what makes the harness's token index and the serve path's `step` the
same address. The safety verdict, though, was established on the CPU
resident decode path: the skip DECISIONS transfer exactly, their safety
does not, because the serve path's routing can differ in fp provenance.

## Consequences

**Claimed.** An expert group can be deleted on the real Metal dispatch
path; the deletion is exactly addressable by `(layer, phase, step)`; the
ledger reports precisely the bytes it avoided; and removing the policy
restores canonical execution bit for bit.

**Not claimed — deliberately.** Any latency result. The report prints a
`PROJECTED` time for the avoided bytes at the arm's own *observed*
streaming rate (not at roofline, which would inflate it) and labels it
"NOT a measured saving". A latency saving is a difference between two
runs, and one run contains one of them. BW-A's own S2/MXFP4 calibration
is the standing reason to distrust byte-to-latency projection here: MXFP4
cut 39.3% of the bytes for 14.7% of the wall, `byte_cut/wall_cut ≈ 2.7x`.

**Also not claimed.** Output quality under any policy. The gate asserts
residual identity for a *skipped* group, which is an arithmetic fact
about the combine, not a statement that skipping is safe for the model.
That question is BW-C's, and its answer is horizon- and
predictability-dependent.

**Known costs.** Three per-layer relaxed atomic operations on the decode
path when disarmed (one load, two adds), far below dispatch noise. The
router still runs for a skipped group — a further saving that is
available later and was left out because it changes what "the semantic
operation stands" means.

## Where BW-B lands

`ExecutionStrategy` is deliberately exhaustive — no `#[non_exhaustive]`,
no catch-all match arms at the dispatch sites. BW-B closed with a
compiled compact-dense representation beating both sparse-gather and
dense across the whole tested range, which is a third answer to the same
question: not "run it" or "delete it" but "run it against a derived
representation". When that has a real materialisation to point at, it
lands as a `CompactDense(..)` variant, and every backend that matches on
this fails to compile until it decides what to do. A `_ => canonical`
fallback would silently serve compact-dense as canonical on whichever
backend nobody remembered to update, and the resulting measurement would
read as a null result rather than an unimplemented one.
