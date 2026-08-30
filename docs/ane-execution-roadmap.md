# LARQL ANE execution — track charter and roadmap

Opened 2026-08-25, branch `worktree-ane-execution`.

The question this track exists to answer:

> LARQL currently uses two of the three compute engines Apple put in
> this machine. Is the third one — the Neural Engine — worth anything
> to an LLM decode path, and if so, for *which* work?

Written so that someone returning cold can pick it up without
re-deriving the environment survey (§2), the pinned geometry (§3), or
the roofline prior (§4).

---

## 1. Status and decision tree

```
rung    subject                                          state
ANE-0   is ANE reachable at all from this machine?       ANSWERED — yes, via Core ML only
ANE-0b  frozen GPU-alone baseline at the ANE-3 shape     BANKED — 288.7/289.1 GB/s raw
ANE-1   does a real decode-shaped op LAND on ANE?        PASS — placed on ANE, 111.9 GB/s equiv
ANE-2   the ANE operating envelope                       DONE — 2A/2B/2C; k rules placement
ANE-3   is ANE throughput additive / recoverable?        CLOSED via 3b — OUTCOME 2, 1.20-1.24x aggregate
ANE-4   wide+shallow draft PLAN on ANE                   synthetic phase CLOSED; ANE-4A real weights NEXT
ANE-5   heterogeneous VINDEX3 partitioning               NOT STARTED
```

```
ANE-1  can we prove placement?
   │ yes
   ▼
ANE-2  is its useful throughput nontrivial?
   │ yes
   ▼
ANE-3  four outcomes (§5)
   ├── 1 hard contention ────────▶ ANE-4 only
   ├── 2 bubble recovery ────────▶ ANE-4 + ANE-5 scoped to the bubbles
   ├── 3 phase complementarity ──▶ ANE-4 + ANE-5 scoped to non-competing ops
   └── 4 additive bandwidth ─────▶ ANE-5 in full; Core AI upgrade justified
```

**ANE-4 does not depend on ANE-3 winning.** A drafter's weight footprint
is small enough that shared-fabric contention may simply be tolerable.
Only outcome 1 kills ANE-5 outright, and it kills nothing else.

Each rung's gate is binary and stated below. ANE-1's gate is *placement
plus a credible steady-state number* — explicitly **not** "faster than
the GPU".

---

## 2. Environment reality (ANE-0, answered 2026-08-25)

```
Model            MacBook Pro, Mac15,8, Apple M3 Max
Cores            12P + 4E, 16-core Neural Engine, 128 GB unified
OS               macOS 15.7.4 (24G517)
Xcode            16.4 (build 16F6)
Active SDK       macOS 15.5
ANE device node  ane0 present, matched and active
coremltools      NOT installed in the default python3
```

Apple's 2026 **Core AI** stack is not available here — it and Metal 4 ML
passes ship with the macOS 26/27-era SDKs, and the active SDK is 15.5.
`ComputeUnitKind.neuralEngine`, stateful specialization, zero-copy Core
AI data paths, custom Metal 4 kernels inside an ML graph: all upgrade
path, none callable today.

What is reachable, and is sufficient for ANE-0b..ANE-4:

```
Core ML (macOS 15 API surface)
  MLModelConfiguration.computeUnits
      .all | .cpuAndNeuralEngine | .cpuAndGPU | .cpuOnly
  MLComputePlan                       per-op device placement, macOS 14.4+
  MLModelAsset / compiled .mlmodelc   specialization + on-disk cache
```

`MLComputePlan` is the load-bearing one: `computeUnits` is a
*preference*, not a placement. A claim that something "ran on ANE"
without a compute plan behind it is not a measurement.

**Core AI deferral is a decision rule, not a preference.** If ANE-3
finds no independent bandwidth, no API can repeal the physical memory
system, and the upgrade buys nothing. If ANE-3 finds headroom, the
upgrade becomes justified *because there is then something worth making
ergonomic*. Either way the decision is downstream of ANE-3.

Before ANE-1 can run, one of:

```
(a) coremltools in a scoped venv       — build the .mlpackage from Python
(b) a small Swift/ObjC harness         — no Python dependency
```

Do (a) for ANE-1. Port to (b) only if ANE-1 and ANE-3 both survive —
the eventual consumer is a Rust crate calling Core ML through objc2.

---

## 3. Pinned geometry (from the real container, not assumed)

Subject model is the same one the CPU programme runs: **Qwen3.8-27B**,
`~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B`, snapshot
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

```
hidden_size            5120
intermediate_size      17408
num_hidden_layers      64      = 16 full_attention + 48 linear_attention
head_dim               256     num_attention_heads 24, num_key_value_heads 4
attn_output_gate       true    -> q_proj output is doubled
linear attn            16 key heads x 128, 48 value heads x 128
dtype                  bfloat16
```

Actual projection shapes, per layer:

```
FFN (all 64 layers)
  gate_proj    5120 -> 17408      89,128,960 params
  up_proj      5120 -> 17408      89,128,960
  down_proj   17408 ->  5120      89,128,960

full attention (16 layers)
  q_proj       5120 -> 12288      (24 heads x 256 x 2, output-gated)
  k_proj       5120 ->  1024
  v_proj       5120 ->  1024
  o_proj       6144 ->  5120

linear attention (48 layers)
  q,k          5120 ->  2048
  v            5120 ->  6144
  o            6144 ->  5120
```

**There is no 5120 x 5120 projection in this model.** The earlier
working figure was wrong; it would have measured a shape LARQL never
executes.

**ANE-1's subject is `5120 -> 17408`** — the gate/up projection. It is
the most repeated large op in the model (128 instances per token before
counting `down_proj`), and FFN is 17.11 G of the ~27.02 GB read per
token, i.e. about 63% of all decode traffic. Getting this one op right
is most of the question.

Traffic and floors for that single op:

```
weights   int8   89.13 MB      f16   178.26 MB
floor at CPU 127 GB/s          0.70 ms   (int8)
floor at GPU 367 GB/s          0.243 ms  (int8)
```

Two consequences that shape ANE-1:

- **One op is 0.24–0.70 ms of work.** That is comfortably above the
  ~44 us the CPU executor spends on a 12-way fan-out, so dispatch
  granularity is not obviously fatal — but Core ML's own per-prediction
  overhead is in the same 0.1–1 ms band as the work itself. ANE-1's
  warm latency must therefore be reported against the traffic floor,
  not just as a number, or submission cost will be silently counted as
  compute.
- **The int8 question is open and it is worth 2x.** Core ML weight
  compression may keep the tensor int8 in memory (89 MB moved) or
  materialise f16 at compute time (178 MB moved). Which one happens
  decides whether ANE is even in the same traffic class as the shipped
  CPU path. Answer it by footprint and implied GB/s in ANE-1; do not
  assume. Note also that Core ML is unlikely to offer int8
  *activations* at all — and CPU-5 established that the activation is
  the binding constraint on this model, not the weight.

---

## 4. The roofline prior

Pre-registered so a positive ANE result cannot be over-read later.

```
CPU  achievable memory bandwidth   ~127 GB/s   (saturates at 2 threads)
GPU  achievable memory bandwidth   ~367 GB/s
     physical fabric ceiling       ~400 GB/s
```

Batch-1 decode over a large dense projection is memory movement, not
arithmetic: the weight tensor is read once and multiplied by one vector.

> **P1.** For a bandwidth-bound batch-1 projection the ANE cannot beat
> the GPU by a large factor — both draw from the same ~400 GB/s fabric
> and the GPU already reaches ~367 of it.

> **P2.** "GPU + ANE concurrently" cannot sum past the fabric ceiling
> *while both are streaming*. But that bounds peak rate, not wall-clock:
> a GPU invocation that streams near the ceiling for only part of its
> duration leaves recoverable time even when no bandwidth is left. So P2
> forbids outcome 4 unless 367 is the GPU's own limit rather than the
> fabric's — and forbids nothing about outcomes 2 and 3. ANE-3 exists to
> separate all four.

> **P3.** ANE's plausible wins are in work that is not weight-traffic
> bound the same way: batched/weight-stationary shapes (the CPU-7B
> regime), a *small* model whose weights stay hot (ANE-4), or
> compute-heavy work overlapped with the GPU's bandwidth-heavy work —
> phase complementarity, not bandwidth addition.

If a measurement appears to beat P1, the first hypothesis is an
artifact — cached weights, a placement that silently fell back to GPU,
or an f16 model believed to be int8 — not a discovery.

---

## 5. The ladder

### ANE-0b — frozen GPU-alone baseline (do this before ANE-1)

Measure and **freeze** the GPU-alone effective bytes/s for *exactly* the
workload ANE-3 will run concurrently: same op, same shape, same dtype,
same duration. Bank it as a file in this branch with the SHA of the code
that produced it.

**The headline ~367 GB/s must not be used as ANE-3's control.** It came
from a different access pattern. Without a same-shape frozen baseline, a
5–10% "gain" at ANE-3 is trivially self-deceiving, and the contention
literature on this machine is unambiguous: A/B bracketing does not
defeat asymmetric contention, and a clean A/B/A/B has already produced a
false +2.4% once. Note also CPU-7A's lesson from the same week: a gate
frozen in ABSOLUTE units inherits a foreign anchor and had to be
recorded malformed rather than moved.

**Wall-clock is the denominator. The floor-adjusted rate is a
diagnostic.** ANE-0b reports both, and they answer different questions:

```
raw effective GB/s        weight_bytes / wall-clock
                          -> THIS is what ANE-3 scores against
floor-adjusted GB/s       weight_bytes / (wall-clock - dispatch floor)
                          -> diagnostic only: what the streaming phase
                             sustains, and therefore how close the GPU
                             already is to the fabric
```

The adjusted number assumes submission and streaming do not overlap,
which is unproven. It must never be used as ANE-3's denominator, and a
high adjusted rate is **not** evidence for outcome 1 — a GPU that
streams near the ceiling for only part of its invocation is precisely
the condition that makes outcome 2 reachable.

**BANKED 2026-08-25** — `bench/baselines/ane/`, sessions `s1-battery`
and `s2-battery`, git `16bedaf2`. On battery by explicit override; ANE-3
must run in the same regime or this must be re-taken.

```
                     s1        s2      spread
raw GB/s   (min)    288.7     289.1     0.15%   <- ANE-3's denominator
floor-adj  (min)    356.1     358.4     0.64%   <- diagnostic only
dispatch floor ms  0.1169    0.1192     1.94%
floor share of isolated wall-clock: 18.9% (min), 24.7% (p50)
ffn_down: 285.1 / 283.3 GB/s at 640 TGs vs gate/up's 2176 -> no
          ROWS_PER_TG geometry surprise
```

Full adjudication, including what this does and does not license:
`bench/baselines/ane/ADJUDICATION.md`.

**The smoke run vindicated the wall-clock rule.** Its floor-adjusted
reading was ~381 GB/s; the banked sessions say ~357. Over the same
interval the raw number moved 293 → 289.

```
                smoke -> banked
raw               293 -> 289      1.4% move
floor-adjusted    381 -> 357      6.3% move
```

The adjustment inherits the floor's variance, and the floor is the
noisier quantity. Had ~381 been banked as "the GPU is at the fabric
ceiling", the follow-up would have been arguing against a number that
had already moved 6%.

**Outcome 2 now has a measured size.** 18.9–24.7% of an isolated
invocation is not weight streaming. That bounds what bubble recovery
could be worth on this op — but it does **not** show the fabric is idle
during that window. `dispatch_floor` prices a minimal single-threadgroup
round trip, not the 2176-threadgroup control's idle time, and the GPU
may be prefetching. Whether the window is usable by another engine is
exactly what ANE-3 measures.

### ANE-1 — does a real decode-shaped op land on ANE?

Deliberately boring and small. **No LARQL integration.**

```
1  real projection geometry            5120 -> 17408, from §3
2  minimal Core ML artifact            one op, f16 first
3  compute-plan placement proof        MLComputePlan, per-op device
4  warm latency distribution           steady state, min-of-N + spread
   cold compile / specialization       measured SEPARATELY, never mixed in
```

**Gate (binary).** ANE placement is proven by compute plan **and** warm
steady-state latency is credible against the §3 traffic floor. Not
"faster than GPU". A rung that only proves placement, with latency
implying an absurd bandwidth, has failed — that pattern means the timing
harness is measuring something other than the op.

**Control before parity.** Before believing any arm, run the instrument
on a case whose answer is known: request ANE placement for an op ANE
cannot host, and confirm the compute plan reports CPU or GPU. If the
harness reports "ANE" for something that cannot run there, every number
it has produced is void.

**Live hypothesis for failure:** a 178 MB f16 constant may simply exceed
what the framework will place on ANE, forcing fallback on size alone. If
so, that is a real and reportable answer, and it points straight at
ANE-4 (small model) rather than at ANE-5.

**RESULT — PASS, banked 2026-08-25** (`bench/ane/`, sessions
`s1-battery` / `s2-battery`, coremltools 9.0). The failure hypothesis did
not occur:

```
5120->17408   requested: CPU+ANE   actual: ANE   (supported: CPU, ANE)

                       s1        s2      spread
latency ms  (min)     1.604     1.582     1.38%
equivalent  (min)     111.1     112.7     1.43%   GB/s
predict floor (min)   0.109     0.106            ~7% of the call
compile (cold)          68 ms             measured separately
```

Both negative controls passed: `CPU_ONLY` requested → reader says CPU,
and `ios16.cumsum` reports `supported: [CPU]` with ANE absent, so the
reader can say "not ANE". Numerical `rel_rms 5.97e-04` against an f64
reference, and — the stronger check — ANE row 8704 = `0.077515` against
ANE-0b's Metal GPU `0.077413`, rel `1.31e-03`: **two independent engines
computing the same projection agree.**

Against the baseline, the ANE is ~2.6x slower than the GPU
(1.593 vs 0.617 ms) and lands near the CPU's ~127 GB/s attainable. That
does not fail the rung — the gate was placement plus a credible
steady-state number, not "faster than the GPU".

**The arithmetic this sets up for ANE-3:**

```
GPU alone   288.9 GB/s
ANE alone   111.9 GB/s
sum         400.8 GB/s   against a ~400 GB/s fabric ceiling
```

The isolated demands sum to almost exactly the fabric. Outcome 4 would
require ~401 GB/s delivered while both stream, which P2 forbids unless
the roofline figure is wrong. Outcomes 1 and 2 are the live hypotheses,
and ANE-0b's ~19% non-streaming window is where outcome 2 would come
from. This is a pre-registration, not ANE-3's answer.

Full detail and the list of what this does **not** license:
`bench/ane/ADJUDICATION.md`.

**One finding for ANE-2 already.** During API bring-up a 512x512 probe
preferred **CPU** while still listing ANE as supported. Placement is
therefore size/shape dependent and the boundary is unmapped — that
boundary is ANE-2's first curve, and it bounds how large an ANE-4
drafter may be.

### ANE-2 — the ANE operating envelope

Not a benchmark number — a small empirical roofline saying where ANE is
good, where it falls back, and where conversion/copy dominates.

```
sweep   matrix shape      the §3 shapes, plus smaller ones to find the fallback edge
        precision         f16, int8-compressed weights, palettized if supported
        batch             1, 2, 4, 8      (the CPU-7B weight-stationary regime)
        fusion            fused sequences where Core ML permits them
report  per cell: placement (compute plan), warm latency, implied GB/s,
        footprint, and whether input/output conversion dominates
```

The batch axis is the one to watch: it is the same axis CPU-7B probes,
and CPU-7B's prediction is that time stays roughly flat to N~4 and then
bends. If ANE bends at N=2, ANE-4 loses most of its value at the same
time CPU-7B would.

**Gate.** Useful throughput at some cell of the envelope is nontrivial —
i.e. within striking distance of the traffic floor, at a shape LARQL
actually runs.

### ANE-3 — is ANE bandwidth additive? (decisive)

```
A     GPU workload alone         (the ANE-0b frozen baseline)
B     ANE workload alone
AB    both issued concurrently
```

**Measure aggregate effective bytes/s, not wall-clock overlap.** Four
outcomes, and they mean entirely different things:

```
1  HARD CONTENTION
   GPU slowdown ~= ANE contribution; aggregate useful throughput flat.
   -> no heterogeneous dense-projection case. ANE-5 dies. ANE-4 survives.

2  BUBBLE RECOVERY
   Peak streaming stays fabric-limited, but concurrent ANE work fills the
   GPU's dispatch / synchronisation / non-streaming periods and WALL-CLOCK
   useful throughput rises.
   -> genuinely useful to LARQL with NO new physical bandwidth discovered.
      Scope ANE-5 to work that fits in the bubbles.

3  PHASE COMPLEMENTARITY
   ANE hosts work whose compute/memory phases don't coincide with GPU
   weight streaming.
   -> asymmetric heterogeneous planning worthwhile; ANE-5 scoped to
      non-competing operators.

4  GENUINE ADDITIVE BANDWIDTH
   GPU stays near its isolated streaming rate while ANE contributes
   meaningful additional traffic.
   -> the big result: the laptop roofline itself changes. ANE-5 becomes
      very interesting, and a Core AI upgrade becomes justified.
```

Outcome 2 is why the taxonomy is four and not three, and it is not
hypothetical here: ANE-0b's own dispatch-floor line measures a
non-streaming region of ~0.14–0.19 ms inside a ~0.61–0.65 ms call. If
the GPU is not occupying the fabric uniformly across the whole
invocation, an end-to-end win is reachable **without** discovering any
new bandwidth. Outcome 2 and outcome 1 are easy to confuse if only peak
streaming rate is reported — which is exactly why the control is banked
as wall-clock.

**Protocol, non-negotiable — this experiment is a known trap:**

```
baseline      the ANE-0b frozen, same-shape GPU number. Not the headline 367.
exclusivity   nothing else may touch GPU or ANE. Verify, don't assume.
              bracketing does NOT defeat asymmetric contention.
interleaving  a clean A/B/A/B has already produced a false +2.4% here.
sessions      two separate sessions, min-of-N within each. Cross-session
              e2e floor is ~±6%; a result inside that band is not a result.
thermal       on AC, thermal state checked before and after. Never on battery.
processes     kill by captured PID, never by pattern.
```

### ANE-4 — drafter on ANE

Survives even a fully negative ANE-3. Put a *small* drafter on the ANE,
keep the 27B target on GPU/CPU, and pay for speculation with silicon
that is otherwise idle rather than with target throughput — inverting
speculative decoding's usual cost.

```
ANE    cheap drafter generates N candidates
GPU    large target verifies N positions in ONE weight traversal
CPU    orchestration, routing, state
```

In order:

```
1  a drafter whose weight traffic is negligible against the target's
   — quantify, do not assume
2  ANE-1/ANE-2 numbers at the DRAFTER's geometry, not the target's
3  acceptance rate on real prompts, measured, not projected
4  target-side batch-N verification — CPU-7C, NOT CPU-7B (see below)
```

Item 4 is the point: this is not an isolated ANE optimisation. It
creates pressure for a primitive the whole engine wants, and the CPU
track produced that primitive independently rather than the ANE track
building a bespoke one. The CPU roadmap already prices the payoff at
~350 ms per 4-position verification with ~2.5 tokens accepted, i.e.
~7 tok/s against today's 4.7 ceiling, and already proposes the drafter
be a cheap VINDEX3 physical plan — which makes the planner answer "how
much model do I need for this purpose".

**What CPU-7B actually licenses, stated precisely.** 7B passed strong on
2026-08-25 — weight-stationary Q8[64] x asym-Q8[16], 5120x5120, six
workers, read from the worse of forward and reversed sweeps:

```
        total cost    per-vector
N=2          1.02x         0.51x
N=4          1.27x         0.32x
N=8          2.41x         0.30x
```

The mechanism is established too: DRAM and cache-resident arms converge
at N=4 and N=8, so batch-1 decode was leaving arithmetic idle behind
memory latency.

But 7B's own adjudication is explicit that it licenses **exactly one
claim — two projection VECTORS are nearly free per weight traversal, at
that shape and operating point.** It does **not** establish that
multi-position verification is free: a real layer also carries
recurrence, norms, activation quantisation, and GatedDelta/attention
state transitions. That is CPU-7C, which is earned but not run.

So ANE-4's target-side prerequisite is **demonstrated for projections
and still open for layers**. Treating 7B as "the target can verify four
tokens for 1.27x" would be a gate–claim congruence violation of the same
class CPU-5 and CPU-7A both produced. Note also that 7B's 5120x5120 is a
synthetic shape — §3 shows the model has no such projection — which is
fine for a mechanism probe and not fine as a layer-level price.

ANE-4 may proceed to items 1–3 now; item 4 waits on 7C.

### ANE-5 — heterogeneous VINDEX3 partitioning

Only on ANE-3 outcome 3. VINDEX3 stays the authority; Core ML becomes
one more lowering target beside CPU and Metal, chosen per-operator
rather than per-model.

Existing constraint to argue against, not around:
`crates/larql-inference/docs/specs/compute-backend-redesign.md` scopes
this out explicitly — "No 'use Metal for attention, Neural Engine for
FFN' routing. One backend per session."

---

## 6. What this track will not build

- **Hand-written ANE kernels.** No such interface exists. Core ML
  decides placement; even in the 2026 stack, custom kernels are Metal
  GPU kernels. ANE work is black-box: construct graph, specialize,
  inspect compute plan, benchmark.
- **A Core ML conversion of the 27B model.** Not first, not second. The
  ladder starts at one op for a reason.
- **Any LARQL integration before ANE-2 passes.**
- **An OS/SDK upgrade to reach Core AI before ANE-3.** See §2.
- **A latency claim from ANE-1.** ANE-1 is a placement proof.

---

## 7. Adjacent state

```
CPU-7    conventional CPU path exhausted at ~4.7 tok/s (264 ms/token,
         81% of attainable). Remaining levers: tokens-per-weight-read
         and fewer bits. CPU-7B is the batch-N weight-stationary probe
         that ANE-4 also needs.
CPU-5    the ACTIVATION is the binding constraint on this model, not the
         weight. Relevant because Core ML likely offers no int8
         activation path at all.
roofline CPU 127, GPU 367, fabric ~400 GB/s, same machine.
VINDEX3  the planner that would own ANE-5's per-operator placement.
```
