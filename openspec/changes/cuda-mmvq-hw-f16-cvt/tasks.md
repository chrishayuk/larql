# cuda-mmvq-hw-f16-cvt — tasks

## 1. Replace software f16 → f32 with PTX cvt

- [x] 1.1 `q4k_mmvq.rs::Q4K_MMVQ_SRC` — replace
      `larql_f16_to_f32` body with one-line inline PTX
      `cvt.f32.f16`.
- [x] 1.2 `q6k_mmvq.rs::Q6K_MMVQ_SRC` — same.

## 2. Tests + bench

- [x] 2.1 All 139 lib tests + 56 integration tests pass with
      no tolerance changes.
- [x] 2.2 `cuda::q4k_mmvq::tests::q4k_mmvq_matches_q4k_direct_on_dequantized_input`
      passes at the existing 1e-3 max-element bound.
- [x] 2.3 `cuda::q6k_mmvq::tests::q6k_mmvq_matches_q6k_f32_on_dequantized_input`
      passes at the existing bound.
- [x] 2.4 Bench (10-run avg, with graph + prefill TC):
      8.04 ms/tok → **7.44 ms/tok (-7.5%, +8.1% tok/s)**.
- [x] 2.5 `larql run` produces identical generated text vs
      pre-change.

## 3. Documentation

- [x] 3.1 `proposal.md` notes the cumulative session impact
      (decode gap 2.18× → 1.71× vs llama.cpp).

## 4. Archive

- [ ] 4.1 Archive when reviewed.
