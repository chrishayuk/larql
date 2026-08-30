# ANE-2C — does int8 change ANE placement policy?

Banked 2026-08-25, battery regime, coremltools 9.0, git `16bedaf2`.
`linear_quantize_weights`, `linear_symmetric`, per-channel, int8.
**Pseudorandom incompressible weights** (normal, seed 20260825) — the
periodic `i % 977` generator used by 2A/2B would have flattered any
compression result and is deliberately not used here.

The three facts are reported independently. None is inferred from
another.

---

## 1. Placement — the lower boundary MOVES, the upper does not

```
                 f16              int8
k =   512        CPU              CPU
k =  1024        CPU              CPU
k =  2048        CPU              CPU
k =  3072        CPU              CPU
k =  4096        CPU              ANE     <- MOVED
k =  4864        CPU              ANE     <- MOVED
k =  4992        ANE              ANE
k =  5120        ANE              ANE
k =  5248        ANE              ANE
k = 16384        ANE              ANE
k = 16512        CPU-only         CPU-only
k = 17408        CPU-only         CPU-only
```

```
lower edge   f16  in (4864, 4992]      int8  in (3072, 4096]
upper edge   f16  in (16384, 16512]    int8  in (16384, 16512]   UNCHANGED
```

**The lower boundary moves down about one step. It does not dissolve.**
k = 512 / 1024 / 2048 / 3072 remain CPU-preferred even at int8, so
genuinely narrow models are still not ANE candidates.

**The upper boundary is precision-independent.** 16384 = 2^14 behaves
identically in both dtypes, and above it ANE is absent from `supported`
in both. This is a hard capability limit on the reduction axis, not a
cost-model preference. **`down_proj` (k = 17408) is not rescued by int8**
— the k-split lowering remains the candidate.

### Fixture control

2A found the f16 lower edge in (4864, 5120] using *periodic* weights.
2C reproduces it at (4864, 4992] using *pseudorandom* weights, and 2C's
f16 latency at k=5120 (0.580 ms) matches 2A's (0.582 ms). **The boundary
is a property of the shape, not of the fixture.**

## 2. Footprint — halves exactly

```
k = 5120     52.44 MB  ->  26.24 MB     2.00x
k = 16384   167.78 MB  ->  83.91 MB     2.00x
k = 17408   178.27 MB  ->  89.15 MB     2.00x
```

Measured from the compiled `weights/weight.bin`, on incompressible
material, so this is real int8 storage rather than a compressible
fixture.

## 3. Latency — improves ~1.15x, NOT 2x

```
                f16 min    int8 min    speedup
k =  4096        0.559       0.445       1.26x   (also changes device)
k =  5120        0.580       0.513       1.13x
k = 16384        1.491       1.274       1.17x
k = 17408        1.539       1.538       1.00x   (CPU-placed both ways)
```

Halving the stored bytes does not halve the time. Read as equivalent
rate over the bytes *actually stored*:

```
k =  5120    f16  90.3 GB/s   ->   int8  51.2 GB/s
k = 16384    f16 112.5 GB/s   ->   int8  65.9 GB/s
```

If int8 halved the traffic, the equivalent rate over stored bytes would
have held roughly constant. It falls by ~40% instead, which is the
signature of **the bytes moved not being the bytes stored** — i.e. a
dequantisation step in the path. Consistent with ANE-1's finding that
the f16 arm stores its full tensor: Core ML compresses on disk, then
widens somewhere before compute.

**int8 is a residency win, not a bandwidth win.** And on CPU-placed
shapes it is not even that: k=17408 is 1.538 vs 1.539 ms.

---

## Consequences

**ANE-4's wide-and-shallow shape SURVIVES, slightly relaxed.** A drafter
still needs k >= ~4096 to be placed on the ANE at all; 1024- and
2048-wide models remain CPU-bound in both precisions. The design remains
"the target's width, a fraction of its depth" — the relaxation is from
~5120 to ~4096, not to arbitrary narrowness.

**One earlier prediction partially flips.** ANE-2A predicted that a
`hidden_size = 4096` model (Llama-7B, Mistral-7B class) would put its
gate/up/q/k/v on CPU. That holds at f16 — and **reverses at int8**,
where k = 4096 is ANE-preferred. Model eligibility is a function of
hidden size *and* weight precision together.

**int8's value for a drafter is residency, not speed.** Halving a
drafter's footprint matters for a machine already holding a 27 GB
target; the ~1.15x latency gain is a bonus, not the argument.

## What this does NOT license

- No claim about where the int8 lower edge sits between 3072 and 4096;
  it was bracketed, not bisected.
- No claim about int8 *activations*. Only weights are quantised here,
  and CPU-5 established the activation is the binding constraint on this
  model.
- No numerical-quality claim. 2C measured placement, footprint and
  latency; it did not check what int8 does to the output. A drafter
  would need that before it could be trusted.
- One session, battery regime.
