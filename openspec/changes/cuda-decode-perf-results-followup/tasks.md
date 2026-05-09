## Documentation only — no code changes

- [x] D.1 Add post-`8.04` checkpoint progression table (this proposal).
- [x] D.2 Document `cuda-mmvq-hw-f16-cvt` win mechanism (-7.5%).
- [x] D.3 Document `cuda-marlin-imma-probe` negative result.
- [x] D.4 Update remaining-gap profile (3.10 ms decode delta vs llama.cpp).
- [x] D.5 Mark Paths A/B/D/E from original retro as closed; mark Path C as open-but-deferred.
- [x] D.6 Cross-reference `cuda-speculative-decoding` as the architectural pivot.
- [x] D.7 Capture session perf model: α=0.6 → 5.15 ms/tok, α=0.7 → 4.39 ms/tok.

## Validation

- [x] V.1 `openspec validate cuda-decode-perf-results-followup --strict` passes.
- [ ] V.2 Cross-link verified by manual review (this proposal references
      `cuda-decode-perf-results`, `cuda-speculative-decoding`,
      `cuda-marlin-imma-probe`, `cuda-mmvq-hw-f16-cvt`,
      `cuda-tensor-cores-q4k`, `cuda-attn-wmma-multi-warp`,
      `cuda-q4k-qkv-fuse-v2`, `cuda-fused-norm-quantize`).
