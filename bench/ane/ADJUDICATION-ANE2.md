# ANE-2 — adjudication (2A and 2B; 2C not yet run)

Banked 2026-08-25, battery regime, coremltools 9.0, git `16bedaf2`.
Built, placed and timed by the **frozen ANE-1 instrument**, imported not
copied. (2B is the one exception: the batch size is part of the model
description, so it needs its own builder; placement and timing still come
from ANE-1.)

---

## 2A — placement is governed by the REDUCTION DEPTH, not by size

The headline. Core ML's device choice for a `linear` tracks **k**, the
input/reduction dimension, and is independent of the output width **n**
and of total bytes.

```
k <= 4864          CPU preferred   (ANE listed as supported)
5120 <= k <= 16384 ANE preferred
k >= 16512         ANE NOT SUPPORTED — absent from `supported` entirely
```

Bytes do not order these. `5120->512` is **5.24 MB on ANE**;
`4096->8192` is **67.11 MB on CPU**. Thirteen times the data, opposite
device.

### The falsifier that earned the claim

Hold k on either side of the edge and vary n across a 16x byte range. If
k is the discriminator, n must not flip the device.

```
k = 4096   n = 512 / 2048 / 8192    ->  CPU, CPU, CPU     n-INDEPENDENT
k = 5120   n = 512 / 2048 / 8192    ->  ANE, ANE, ANE     n-INDEPENDENT
```

### The two edges

```
lower   4096->5120  CPU      4864->5120  CPU      5120->5120  ANE
upper  16384->5120  ANE     16512->5120  CPU-only  17408->5120  CPU-only
```

16384 = 2^14, and the upper edge is a **capability** limit, not a
preference: ANE disappears from `supported`. Output width does not move
it — `17408->512` (17.83 MB) is equally CPU-only.

### What this means for LARQL

Every projection in Qwen3.8-27B is ANE-eligible **except one**:

```
gate_proj / up_proj   5120 -> 17408    k =  5120   ANE
q_proj                5120 -> 12288    k =  5120   ANE
k_proj / v_proj       5120 ->  1024    k =  5120   ANE
o_proj                6144 ->  5120    k =  6144   ANE
linear q/k, v         5120 -> 2048/6144 k = 5120   ANE
down_proj            17408 ->  5120    k = 17408   NOT SUPPORTED
```

**`down_proj` cannot run on the ANE in this form.** It is a third of FFN
traffic and ~21% of the whole model's bytes per token. A layer cannot be
placed wholly on the ANE.

There is an obvious lowering that VINDEX3 is exactly the right layer to
own: split `down_proj` along k into two halves of 8704 (both under
16384) and sum the partials. Untested — it is a follow-up, not a claim.

**And note how close gate/up sits to the lower edge.** k = 5120 clears a
threshold somewhere in (4864, 5120]. A model with `hidden_size = 4096` —
Llama-7B, Mistral-7B, and many others — would have k = 4096 for its
gate/up/q/k/v and would land on **CPU**, while its `down_proj`
(k = 11008, under 16384) would land on **ANE**: the exact inverse of
Qwen3.8. ANE eligibility is a function of the model's hidden size. This
is a prediction from the k-rule, not a measurement.

### Small-end behaviour

Every square shape tested prefers CPU, up to 3072x3072 (18.87 MB):

```
256^2  512^2  768^2  1024^2  1536^2  2048^2  3072^2   -> all CPU
```

Consistent with the k-rule: all have k < 4864. The equivalent rate on
these cells is not memory traffic — 0.13-18.87 MB is cache-resident, and
`2048->2048` reads 196 GB/s for that reason.

**The smallest useful unit of ANE work is not a size, it is a width.**

---

## 2B — batching is nearly free on the ANE

Same shape ANE-1 proved, `5120 -> 17408`, N vectors through one weight
traversal.

```
   N  device   min ms   T(N)/T(1)  per-vector    CPU-7B ref
   1     ANE    1.597        1.00        1.00          1.00
   2     ANE    1.602        1.00        0.50          1.02
   4     ANE    1.616        1.01        0.25          1.27
   8     ANE    1.660        1.04        0.13          2.41
```

**Eight vectors for 4% more time.** The ANE is almost entirely
weight-traffic-bound at batch 1 and has enormous idle arithmetic
capacity — the same mechanism CPU-7B found on the CPU, but far more
pronounced: the CPU's curve has bent to 2.41x by N=8 where the ANE's is
still 1.04x.

N=1 through this builder reproduces ANE-1's banked 1.58-1.60 ms, so the
batched build is consistent with the frozen instrument.

The CPU-7B column is a reference point, not a control: different engine,
different representation (Q8[64] x asym-Q8[16], 5120x5120).

### A measurement this does NOT license, and it matters

Comparing **ANE at N=8** (0.208 ms/vector) against **GPU at N=1**
(0.617 ms/vector) would be unfair and is not claimed here. ANE-0b
measured only N=1, because `f16_gemv` is a gemv — the GPU would also
amortise a batched traversal, and by how much is **unmeasured**.

> **Owed: a GPU batch-N arm at the same shape.** Until it exists, "the
> ANE beats the GPU when batched" is not a supported statement. What is
> supported is: *on the ANE, batching to 8 costs 4%.*

---

## Consequences for ANE-4, which now has a concrete shape

Two of 2A's findings pull against each other and resolve into a design:

- A drafter must be **wide** (k >= ~5120) or it will not be placed on
  the ANE at all. Narrow small models — a 1024-hidden draft model — run
  on CPU.
- A drafter can be **shallow**: depth costs invocations, not placement.

So the ANE-4 drafter is *not* "a small model". It is **the target's
width with a fraction of its depth** — which is exactly the "cheap
VINDEX3 physical plan: fewer layers, lower precision" the CPU roadmap
already proposed, arriving now as a hardware constraint rather than a
preference.

2B then says such a drafter can propose several tokens per weight
traversal at almost no extra cost.

---

## Still open

- **2C (int8)** not run. Three separate questions — placement,
  footprint, latency — which need not move together. Must use
  pseudorandom weight material: the `i % 977` generator used here is
  periodic and would flatter any compression result.
- **GPU batch-N arm**, per above.
- **`down_proj` k-split** as a lowering, untested.
- Everything here is one session. 2A/2B are placement-dominated findings
  and placement is deterministic, but the *timings* have not had the
  two-session treatment ANE-0b and ANE-1 got.
