## Context

The existing CUDA Q4 implementation closed the correctness gap by reusing CPU
dequantizers, uploading a temporary f32 matrix, and running cuBLAS GEMV. That
is acceptable for parity tests but bad for decode. A Gemma 3 4B decode step
touches many Q4_K matrices per token; repeatedly expanding each matrix to f32
on the host dominates the observed benchmark.

The first benchmark pass against `output/gemma-3-4b-it-vindex` showed the
problem clearly: GPU forward averaged roughly 7.76 s/token and LM head roughly
1.53 s/token. Both paths currently pay quantized-weight expansion costs.

## Goals / Non-Goals

**Goals:**

- Implement a direct CUDA Q4_K matvec kernel for packed Q4_K weights.
- Make the direct path the default for `CudaBackend::q4k_matvec`.
- Keep the previous host-dequant path callable as a debugging fallback.
- Exercise the direct path in real decode where Q4_K weights are dispatched via
  CUDA quant matvec helpers.
- Complete a real `larql bench --backends cuda` pass and record the result.

**Non-Goals:**

- Port all ggml quant formats in this change.
- Implement permanent whole-model GPU residency in one step.
- Replace cuBLAS f32 paths.
- Solve QKV projection residency if it requires a different fused projection
  contract than Q4_K matvec.

## Decisions

### D1 - Direct kernel first, full residency later

The direct Q4_K kernel will accept packed host bytes and `x: &[f32]` at the
backend API boundary, then upload the packed bytes and input vector to the GPU
for one launch. This still copies packed weights per call, but packed Q4_K is
much smaller than expanded f32 and avoids CPU dequant entirely.

Full persistent device residency should come after this because it changes
lifetime management and vindex loading contracts. The direct kernel gives a
smaller, measurable slice and remains useful once residency is added.

### D2 - Preserve host-dequant fallback

The existing implementation is the correctness oracle. The new direct path will
be bypassable with an environment variable so failures can be isolated without
removing CUDA support from the bench.

### D3 - Keep the test oracle CPU-based

Tests compare direct CUDA output to the CPU quant matvec/dequant reference. The
kernel should match within the existing Q4 tolerance instead of asserting exact
bit identity, because accumulation order differs from the host path.

## Risks / Trade-offs

- **Risk: Direct-but-not-resident still copies weights per call.** Mitigation:
  packed Q4_K upload is far smaller than f32 upload and proves the kernel
  before introducing device-resident model lifetime.
- **Risk: First kernel underperforms optimized ggml CUDA.** Mitigation: keep
  acceptance based on material improvement over the current 9.25 s/token
  baseline, then profile the next bottleneck.
- **Risk: Q4_K block layout bugs are hard to spot at large shapes.** Mitigation:
  add small deterministic tests plus production-like parity tests.
- **Risk: Decode has multiple quant formats.** Mitigation: route Q4_K first and
  keep existing fallback for other formats.
