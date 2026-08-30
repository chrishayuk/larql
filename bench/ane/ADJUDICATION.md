# ANE-1 — adjudication

Banked 2026-08-25. Sessions `s1-battery`, `s2-battery`, git `16bedaf2`,
coremltools 9.0, macOS 15.7.4, **battery regime** (matching ANE-0b).

Subject: Qwen3.8-27B FFN gate/up, `5120 -> 17408`, f16, 178.26 MB.
One op. No int8.

## Verdict: PASS — the projection is placed on the Neural Engine

```
5120->17408   requested: CPU+ANE   actual: ANE   (supported: CPU, ANE)
```

`preferred_compute_device` is `MLNeuralEngineComputeDevice`. This is the
placement, not merely a capability: `supported` says the device *could*
host the op, `preferred` says it *will*.

The pre-registered failure mode — Core ML refusing to specialise a
178 MB f16 constant for ANE — **did not occur**.

## Controls

| control | result |
|---|---|
| **3a. Reader tracks the request.** Same model, `CPU_ONLY` requested. | Reader reports CPU. **PASS** — it does not report ANE by habit. |
| **3b. ANE-unsupported op.** `ios16.cumsum` under `CPU_AND_NE`. | `supported: [CPU]`, ANE absent. **PASS** — the reader can say "not ANE". |
| **4. Numerical.** Full 17408-wide output vs an f64 reference from the same f16 weights. | `rel_rms 5.970e-04`. **PASS** |
| **4b. Cross-instrument.** Same weights and activation as ANE-0b. | ANE row 8704 = `0.077515` vs the Metal GPU's banked `0.077413`, rel `1.31e-03`. Two independent engines, same projection, same answer. |

## Latency and equivalent rate

```
                       s1        s2      spread
latency ms  (min)     1.604     1.582     1.38%
latency ms  (p50)     1.709     1.695     0.82%
equivalent  (min)     111.1     112.7     1.43%   GB/s
equivalent  (p50)     104.3     105.2     0.86%   GB/s
predict floor (min)   0.109     0.106
predict floor (p50)   0.142     0.138
compile (cold)          68 ms     ~70 ms   measured separately, never mixed in
```

Floor is ~7% of the ANE call, against 19% for the GPU's — the ANE
invocation is far more dominated by work than by call overhead.

**"Equivalent rate" is literal.** It is what the latency implies *if*
178.26 MB were fetched per prediction. One piece of evidence that it
roughly is: the compiled `weights/weight.bin` is 178,292,928 bytes — the
full uncompressed f16 tensor, no palettisation or compression applied.
178 MB also cannot sit in any ANE-local memory. But this is not yet a
measurement of physical traffic; that is ANE-2's job.

## Against the ANE-0b baseline

```
              latency ms      equivalent GB/s
GPU (ANE-0b)      0.617              288.9
ANE (ANE-1)       1.593              111.9
ratio             2.58x               0.39x
```

The ANE is ~2.6x slower than the GPU on this op, and lands close to the
CPU's ~127 GB/s attainable rate. **This does not fail ANE-1**: the gate
was placement plus a physically credible steady-state number, explicitly
not "faster than the GPU".

## The arithmetic that now sets up ANE-3

```
GPU alone   288.9 GB/s
ANE alone   111.9 GB/s
sum         400.8 GB/s   against a ~400 GB/s fabric ceiling
```

The two engines' isolated demands sum to almost exactly the machine's
fabric ceiling. That makes ANE-3 unusually well posed: outcome 4
(genuine additive bandwidth) requires the fabric to deliver ~401 GB/s
while both stream, which P2 forbids unless the roofline figure is wrong.
Outcomes 1 and 2 are the live hypotheses, and the ~19% non-streaming
window ANE-0b measured is where outcome 2 would come from.

Do not treat this as ANE-3's answer. It is a pre-registration.

## First-order ANE-4 sizing

At ~112 GB/s equivalent, a drafter that must produce a token in ~2 ms
can move roughly 225 MB of weights per token. That is a real constraint
on drafter size, and it is a first-order estimate from one shape — the
envelope (ANE-2) is what should actually price it.

## What this does NOT license

- **No claim about int8.** f16 only, deliberately.
- **No claim about measured physical traffic.** See above.
- **No claim about AC behaviour.** Battery, by override, to match ANE-0b.
- **No claim that ANE placement generalises across shapes.** A 512x512
  probe during API bring-up preferred **CPU** despite listing ANE as
  supported — so placement is size- and shape-dependent, and the
  boundary is unmapped. That is exactly ANE-2.
- **Weight-pattern caveat.** The synthetic weights are periodic
  (`i % 977`) and therefore highly compressible. Nothing compressed them
  here, but any future palettised or compressed arm must not use this
  generator, or it will flatter itself.
