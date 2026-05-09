# cuda-fused-norm-add — tasks

## 1. Fused kernel

- [x] 1.1 `RMS_NORM_ADD_SRC` PTX in `cuda::elem` with the
      `rms_norm_add_f32` kernel. Single-block, shared-memory
      reduction; `scale` arg folds in `layer_scalar`.
- [x] 1.2 `RMS_NORM_ADD_FUNC: OnceLock` cell + lazy
      `compile_ptx + load_module + load_function`.

## 2. Wrapper

- [x] 2.1 `elem::rms_norm_add_device(dst, src, weight, n, eps,
      norm_offset, scale)` Rust wrapper. Mirrors the existing
      `rms_norm_device_into` calling convention but takes
      `dst` as `&mut CudaSlice<f32>` (in-place add) instead of
      a fresh-allocation output.

## 3. Pipeline integration

- [x] 3.1 Replace the post-attn `rms_norm_device_into +
      add_in_place_device` pair in
      `decode::run_decode_pipeline_into_scratch` with one
      `rms_norm_add_device` call. Drops the
      `scratch.attn_normed` write+read.
- [x] 3.2 Replace the post-ffn fusion AND fold the optional
      `scale_inplace_device(layer_scalar)` into the kernel's
      `scale` arg. Drops `scratch.ffn_normed` write+read AND
      one launch when `layer_scalar != 1.0`.

## 4. Tests + bench

- [x] 4.1 `decode_token_phase1_matches_host_fallback` passes.
- [x] 4.2 `decode_token_graph_matches_per_call_over_5_steps`
      passes.
- [x] 4.3 10-run bench: 8.19 ms/tok avg (excluding 1 outlier:
      8.09). Pre-change 8.23 ms/tok.

## 5. Documentation + archive

- [x] 5.1 `proposal.md` notes the architectural rationale
      (TensorRT-LLM `RMSNormPlugin`-style residual fusion) and
      why the gain is modest at short context.
- [ ] 5.2 Archive when reviewed.
