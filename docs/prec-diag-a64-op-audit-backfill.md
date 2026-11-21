# PREC-DIAG A64 operation-level dtype audit: backfill (diagnostic)

**Diagnostic only. No sealed verdict is changed or re-evaluated: no sealed run is read, no accuracy metric is computed, and the forwards are throwaway (at most 16 tokens; logits discarded unread).**

Issue #897. Audit schema `larql.prec-diag-1.a64-op-audit/1`. Run 2026-11-21 on arm64, torch 2.14.0, transformers 5.5.0, 8 threads. The full record is the JSON beside this file.

Each subject is one A64 forward made as `hf_oracle.forward` makes it, with the existing patches and the module-boundary hooks, plus audit part 5 (`op_audit.py`).

## Findings

- **gemma3_text** (2 subjects): part 5 passes on the A64 forward; 0 non-float64 op events; every op with floating tensors is classified; 48 kernel probes, 0 failed. Dtype conversions: torch.int64 -> torch.float64 at `hf_oracle.py:502 a64_patches.<locals>.rot`.
- **qwen3** (2 subjects): part 5 passes on the A64 forward; 0 non-float64 op events; every op with floating tensors is classified; 46 kernel probes, 0 failed. Dtype conversions: torch.int64 -> torch.float64 at `hf_oracle.py:502 a64_patches.<locals>.rot`.
- **The gap #897 names, on these families**: with A64's patches removed, part 5 fails on 4 of 4 subjects while the module-boundary audit (part 2) reports 0 problems on the same forward. The shipped float32 sites (RMSNorm, rotary, eager softmax) cast and cast back inside one call. Part 3 (patched-function call counts and witnesses) also fails on 4 of 4 control forwards, because a removed patch is never called; it can only see a site that has a patch, while part 5 needs none.
- **Gemma 3's scaled embedding is float64 work unpatched**: on 2 of 2 Gemma 3 subjects the `embed_scale` buffer holds sqrt(hidden) in float64, not a float32-rounded value (`from_pretrained(dtype=float64)` builds it in float64; recorded in `control_unpatched.embed_scale`). So the control, which removed the scaled-embedding patch, correctly shows no embedding site: there is no float32 work there under A64's load. Part 5 can still not see a float64 tensor holding a float32-rounded value; none was found here.
- Sealed verdicts are unchanged. This backfill re-evaluates none of them.

## Summary

| subject | family | tokens | ops traced | floating ops | non-f64 ops | violation sites | unclassified | probes (passed) | casts (non-f64) | part 5 | parts 2-4 / module-boundary problems |
|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic gemma3_text (test_prec_diag_hf_oracle) | gemma3_text | 24 | 793 | 708 | 0 | 0 | 0 | 24 (24) | 4 (0) | pass | pass / 0 |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | qwen3 | 24 | 480 | 441 | 0 | 0 | 0 | 23 (23) | 1 (0) | pass | pass / 0 |
| google/gemma-3-270m-it | gemma3_text | 15 | 3187 | 3060 | 0 | 0 | 0 | 24 (24) | 18 (0) | pass | pass / 0 |
| Qwen/Qwen3-0.6B | qwen3 | 14 | 3805 | 3766 | 0 | 0 | 0 | 23 (23) | 1 (0) | pass | pass / 0 |

## Non-float64 operations, by site

None: in every subject, every op read, returned and wrote float64 only, and every op with floating tensors was classified.

## Dtype conversions, by site

Every conversion the forward made, faulting or not.

| subject | site | op | from | to | count | non-f64 |
|---|---|---|---|---|---|---|
| synthetic gemma3_text (test_prec_diag_hf_oracle) | `hf_oracle.py:502 a64_patches.<locals>.rot` | aten._to_copy.default | torch.int64 | torch.float64 | 4 | no |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | `hf_oracle.py:502 a64_patches.<locals>.rot` | aten._to_copy.default | torch.int64 | torch.float64 | 1 | no |
| google/gemma-3-270m-it | `hf_oracle.py:502 a64_patches.<locals>.rot` | aten._to_copy.default | torch.int64 | torch.float64 | 18 | no |
| Qwen/Qwen3-0.6B | `hf_oracle.py:502 a64_patches.<locals>.rot` | aten._to_copy.default | torch.int64 | torch.float64 | 1 | no |

## Kernel probes

Accumulating ops are probed at every distinct shape and stride the forward used; `error` must be 0 (exact) or within 1e-12, and the float32 counterfactual must miss.

| subject | op | reduction length | comparison | error | f32 counterfactual | pass |
|---|---|---|---|---|---|---|
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten._softmax.default | 24 | relative | 0.0 | 9.750947551770755e-07 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.add.Tensor |  | exact | 0.0 | 4.732177059096898e-07 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.bmm.default | 32 | exact | 0.0 | 7.450581374079945e-08 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.bmm.default | 24 | exact | 0.0 | 7.450581196444261e-08 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.cos.default |  | mixed | 1.1102230246251565e-16 | 0.00012141299540856243 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.div.Tensor |  | exact | 0.0 | 9.096925782614562e-05 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.gelu.default |  | mixed | 2.2041830166630305e-16 | 1.5411372041282746e-07 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.mean.dim | 80 | exact | 0.0 | 5.587937647322239e-10 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.mean.dim | 32 | exact | 0.0 | 1.862645364336668e-09 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.mean.dim | 32 | exact | 0.0 | 1.862645364336668e-09 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.mm.default | 80 | exact | 0.0 | 7.450582439894049e-08 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.mm.default | 80 | exact | 0.0 | 7.450582439894049e-08 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.mm.default | 64 | exact | 0.0 | 7.450582084622681e-08 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.mm.default | 80 | exact | 0.0 | 7.450582439894049e-08 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.mm.default | 96 | exact | 0.0 | 7.450582795165417e-08 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.mm.default | 80 | exact | 0.0 | 7.450582439894049e-08 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.mul.Tensor |  | exact | 0.0 | 1.1624123743558812e-06 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.neg.default |  | exact | 0.0 | 1.1904007646279524e-07 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.pow.Scalar |  | relative | 2.0841300435571533e-16 | 3.1717929890127353e-07 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.pow.Scalar |  | relative | 2.1109516822708242e-16 | 4.470631667412199e-07 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.pow.Tensor_Scalar |  | relative | 1.9173745329607644e-16 | 1.667059936045557e-07 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.reciprocal.default |  | exact | 0.0 | 0.0001011114586617623 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.rsqrt.default |  | relative | 0.0 | 1.0166619836946932e-07 | pass |
| synthetic gemma3_text (test_prec_diag_hf_oracle) | aten.sin.default |  | mixed | 1.1102230246251565e-16 | 0.0001206728512435154 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten._softmax.default | 24 | relative | 0.0 | 9.750947551770755e-07 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.add.Tensor |  | exact | 0.0 | 4.732177059096898e-07 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.bmm.default | 32 | exact | 0.0 | 7.450581374079945e-08 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.bmm.default | 24 | exact | 0.0 | 7.450581196444261e-08 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.cos.default |  | mixed | 1.1102230246251565e-16 | 0.00012141299540856243 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.div.Tensor |  | exact | 0.0 | 9.096925782614562e-05 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.mean.dim | 64 | exact | 0.0 | 9.313227931906365e-10 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.mean.dim | 32 | exact | 0.0 | 1.862645364336668e-09 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.mean.dim | 32 | exact | 0.0 | 1.862645364336668e-09 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.mm.default | 64 | exact | 0.0 | 7.450582084622681e-08 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.mm.default | 64 | exact | 0.0 | 7.450582084622681e-08 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.mm.default | 128 | exact | 0.0 | 7.450583505708153e-08 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.mm.default | 64 | exact | 0.0 | 7.450582084622681e-08 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.mm.default | 96 | exact | 0.0 | 7.450582795165417e-08 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.mm.default | 64 | exact | 0.0 | 7.450582084622681e-08 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.mul.Tensor |  | exact | 0.0 | 1.1624123743558812e-06 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.neg.default |  | exact | 0.0 | 1.1904007646279524e-07 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.pow.Scalar |  | relative | 2.1109516822708242e-16 | 4.470631667412199e-07 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.pow.Tensor_Scalar |  | relative | 1.9173745329607644e-16 | 1.667059936045557e-07 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.reciprocal.default |  | exact | 0.0 | 0.0001011114586617623 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.rsqrt.default |  | relative | 0.0 | 1.0166619836946932e-07 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.silu.default |  | mixed | 2.597796577926198e-16 | 1.4295600100957658e-07 | pass |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | aten.sin.default |  | mixed | 1.1102230246251565e-16 | 0.0001206728512435154 | pass |
| google/gemma-3-270m-it | aten._softmax.default | 15 | relative | 0.0 | 9.667128439451436e-07 | pass |
| google/gemma-3-270m-it | aten.add.Tensor |  | exact | 0.0 | 4.732177059096898e-07 | pass |
| google/gemma-3-270m-it | aten.bmm.default | 256 | exact | 0.0 | 7.450586347879096e-08 | pass |
| google/gemma-3-270m-it | aten.bmm.default | 15 | exact | 0.0 | 7.450580996604117e-08 | pass |
| google/gemma-3-270m-it | aten.cos.default |  | mixed | 1.1102230246251565e-16 | 0.00012141299540856243 | pass |
| google/gemma-3-270m-it | aten.div.Tensor |  | exact | 0.0 | 9.096925782614562e-05 | pass |
| google/gemma-3-270m-it | aten.gelu.default |  | mixed | 2.2041830166630305e-16 | 1.5411372041282746e-07 | pass |
| google/gemma-3-270m-it | aten.mean.dim | 640 | exact | 0.0 | 6.984941488055729e-11 | pass |
| google/gemma-3-270m-it | aten.mean.dim | 256 | exact | 0.0 | 2.328308648311128e-10 | pass |
| google/gemma-3-270m-it | aten.mean.dim | 256 | exact | 0.0 | 2.328308648311128e-10 | pass |
| google/gemma-3-270m-it | aten.mm.default | 640 | exact | 0.0 | 7.450594874391925e-08 | pass |
| google/gemma-3-270m-it | aten.mm.default | 640 | exact | 0.0 | 7.450594874391925e-08 | pass |
| google/gemma-3-270m-it | aten.mm.default | 1024 | exact | 0.0 | 7.450603400904754e-08 | pass |
| google/gemma-3-270m-it | aten.mm.default | 640 | exact | 0.0 | 7.450594874391925e-08 | pass |
| google/gemma-3-270m-it | aten.mm.default | 2048 | exact | 0.0 | 7.450626138272298e-08 | pass |
| google/gemma-3-270m-it | aten.mm.default | 640 | exact | 0.0 | 7.450594874391925e-08 | pass |
| google/gemma-3-270m-it | aten.mul.Tensor |  | exact | 0.0 | 1.1624123743558812e-06 | pass |
| google/gemma-3-270m-it | aten.neg.default |  | exact | 0.0 | 1.1904007646279524e-07 | pass |
| google/gemma-3-270m-it | aten.pow.Scalar |  | relative | 2.0841300435571533e-16 | 3.1717929890127353e-07 | pass |
| google/gemma-3-270m-it | aten.pow.Scalar |  | relative | 2.1109516822708242e-16 | 4.470631667412199e-07 | pass |
| google/gemma-3-270m-it | aten.pow.Tensor_Scalar |  | relative | 1.9173745329607644e-16 | 1.667059936045557e-07 | pass |
| google/gemma-3-270m-it | aten.reciprocal.default |  | exact | 0.0 | 0.0001011114586617623 | pass |
| google/gemma-3-270m-it | aten.rsqrt.default |  | relative | 0.0 | 1.0166619836946932e-07 | pass |
| google/gemma-3-270m-it | aten.sin.default |  | mixed | 1.1102230246251565e-16 | 0.0001206728512435154 | pass |
| Qwen/Qwen3-0.6B | aten._softmax.default | 14 | relative | 0.0 | 9.657815205699532e-07 | pass |
| Qwen/Qwen3-0.6B | aten.add.Tensor |  | exact | 0.0 | 4.732177059096898e-07 | pass |
| Qwen/Qwen3-0.6B | aten.bmm.default | 128 | exact | 0.0 | 7.450583505708153e-08 | pass |
| Qwen/Qwen3-0.6B | aten.bmm.default | 14 | exact | 0.0 | 7.450580974399656e-08 | pass |
| Qwen/Qwen3-0.6B | aten.cos.default |  | mixed | 1.1102230246251565e-16 | 0.00012141299540856243 | pass |
| Qwen/Qwen3-0.6B | aten.div.Tensor |  | exact | 0.0 | 9.096925782614562e-05 | pass |
| Qwen/Qwen3-0.6B | aten.mean.dim | 1024 | exact | 0.0 | 5.82078827412319e-11 | pass |
| Qwen/Qwen3-0.6B | aten.mean.dim | 128 | exact | 0.0 | 4.656615076176207e-10 | pass |
| Qwen/Qwen3-0.6B | aten.mean.dim | 128 | exact | 0.0 | 4.656615076176207e-10 | pass |
| Qwen/Qwen3-0.6B | aten.mm.default | 1024 | exact | 0.0 | 7.450603400904754e-08 | pass |
| Qwen/Qwen3-0.6B | aten.mm.default | 1024 | exact | 0.0 | 7.450603400904754e-08 | pass |
| Qwen/Qwen3-0.6B | aten.mm.default | 2048 | exact | 0.0 | 7.450626138272298e-08 | pass |
| Qwen/Qwen3-0.6B | aten.mm.default | 1024 | exact | 0.0 | 7.450603400904754e-08 | pass |
| Qwen/Qwen3-0.6B | aten.mm.default | 3072 | exact | 0.0 | 7.450648875639843e-08 | pass |
| Qwen/Qwen3-0.6B | aten.mm.default | 1024 | exact | 0.0 | 7.450603400904754e-08 | pass |
| Qwen/Qwen3-0.6B | aten.mul.Tensor |  | exact | 0.0 | 1.1624123743558812e-06 | pass |
| Qwen/Qwen3-0.6B | aten.neg.default |  | exact | 0.0 | 1.1904007646279524e-07 | pass |
| Qwen/Qwen3-0.6B | aten.pow.Scalar |  | relative | 2.1109516822708242e-16 | 4.470631667412199e-07 | pass |
| Qwen/Qwen3-0.6B | aten.pow.Tensor_Scalar |  | relative | 1.9173745329607644e-16 | 1.667059936045557e-07 | pass |
| Qwen/Qwen3-0.6B | aten.reciprocal.default |  | exact | 0.0 | 0.0001011114586617623 | pass |
| Qwen/Qwen3-0.6B | aten.rsqrt.default |  | relative | 0.0 | 1.0166619836946932e-07 | pass |
| Qwen/Qwen3-0.6B | aten.silu.default |  | mixed | 2.597796577926198e-16 | 1.4295600100957658e-07 | pass |
| Qwen/Qwen3-0.6B | aten.sin.default |  | mixed | 1.1102230246251565e-16 | 0.0001206728512435154 | pass |

## Probe memory

The largest single probe's own allocation, and the process peak RSS before and after the probes (the model and logits are live throughout).

| subject | matmuls / nn.Linear | largest probe (bytes) | peak RSS before probes | after probes |
|---|---|---|---|---|
| synthetic gemma3_text (test_prec_diag_hf_oracle) | 29 / 29 | 264960 | 804077568 | 804077568 |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | 22 / 22 | 223488 | 804077568 | 804077568 |
| google/gemma-3-270m-it | 127 / 127 | 1373711360 | 4661673984 | 4661673984 |
| Qwen/Qwen3-0.6B | 197 / 197 | 1261791232 | 8807071744 | 8807071744 |

## Control: the same forward with every A64 patch removed

Not an A64 forward: transformers' modeling file as shipped, on the same float64 weights. It shows that part 5 sees the families' own float32 sites on these very subjects, and that the module-boundary audit (part 2) does not.

| subject | patches removed | non-f64 ops / floating ops | part 5 | module-boundary problems |
|---|---|---|---|---|
| synthetic gemma3_text (test_prec_diag_hf_oracle) | embedding, rmsnorm, rotary, softmax | 330 / 688 | FAIL | 0 |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | rmsnorm, rotary, softmax | 117 / 400 | FAIL | 0 |
| google/gemma-3-270m-it | embedding, rmsnorm, rotary, softmax | 1450 / 3012 | FAIL | 0 |
| Qwen/Qwen3-0.6B | rmsnorm, rotary, softmax | 892 / 3425 | FAIL | 0 |

| subject | part 3 on the control | part 3 problems (first) | embed_scale |
|---|---|---|---|
| synthetic gemma3_text (test_prec_diag_hf_oracle) | FAIL | patched-function call counts {'rmsnorm': 0, 'rotary': 0, 'softmax': 0, 'embedding': 0} != {'rmsnorm': 25, 'rotary': 4, 'softmax': 4, 'embedding': 1} | torch.float64 8.94427190999916; float32-rounded: False; float64 sqrt: True |
| synthetic qwen3 (test_prec_diag_hf_oracle_qwen3) | FAIL | patched-function call counts {'rmsnorm': 0, 'rotary': 0, 'softmax': 0} != {'rmsnorm': 13, 'rotary': 1, 'softmax': 3} | n/a |
| google/gemma-3-270m-it | FAIL | patched-function call counts {'rmsnorm': 0, 'rotary': 0, 'softmax': 0, 'embedding': 0} != {'rmsnorm': 109, 'rotary': 18, 'softmax': 18, 'embedding': 1} | torch.float64 25.298221281347036; float32-rounded: False; float64 sqrt: True |
| Qwen/Qwen3-0.6B | FAIL | patched-function call counts {'rmsnorm': 0, 'rotary': 0, 'softmax': 0} != {'rmsnorm': 113, 'rotary': 1, 'softmax': 28} | n/a |

Non-float64 sites in the control:

- synthetic gemma3_text (test_prec_diag_hf_oracle): `modeling_gemma3.py:139 Gemma3RMSNorm._norm` (250 op events): `return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)`
- synthetic gemma3_text (test_prec_diag_hf_oracle): `modeling_gemma3.py:142 Gemma3RMSNorm.forward` (50 op events): `output = self._norm(x.float())`
- synthetic gemma3_text (test_prec_diag_hf_oracle): `modeling_gemma3.py:145 Gemma3RMSNorm.forward` (150 op events): `output = output * (1.0 + self.weight.float())`
- synthetic gemma3_text (test_prec_diag_hf_oracle): `modeling_gemma3.py:146 Gemma3RMSNorm.forward` (25 op events): `return output.type_as(x)`
- synthetic gemma3_text (test_prec_diag_hf_oracle): `modeling_gemma3.py:220 Gemma3RotaryEmbedding.forward` (16 op events): `inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)`
- synthetic gemma3_text (test_prec_diag_hf_oracle): `modeling_gemma3.py:221 Gemma3RotaryEmbedding.forward` (8 op events): `position_ids_expanded = position_ids[:, None, :].float()`
- synthetic gemma3_text (test_prec_diag_hf_oracle): `modeling_gemma3.py:225 Gemma3RotaryEmbedding.forward` (56 op events): `freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)`
- synthetic gemma3_text (test_prec_diag_hf_oracle): `modeling_gemma3.py:226 Gemma3RotaryEmbedding.forward` (8 op events): `emb = torch.cat((freqs, freqs), dim=-1)`
- synthetic gemma3_text (test_prec_diag_hf_oracle): `modeling_gemma3.py:227 Gemma3RotaryEmbedding.forward` (16 op events): `cos = emb.cos() * attention_scaling`
- synthetic gemma3_text (test_prec_diag_hf_oracle): `modeling_gemma3.py:228 Gemma3RotaryEmbedding.forward` (16 op events): `sin = emb.sin() * attention_scaling`
- synthetic gemma3_text (test_prec_diag_hf_oracle): `modeling_gemma3.py:230 Gemma3RotaryEmbedding.forward` (8 op events): `return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)`
- synthetic gemma3_text (test_prec_diag_hf_oracle): `modeling_gemma3.py:305 eager_attention_forward` (20 op events): `attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)`
- synthetic qwen3 (test_prec_diag_hf_oracle_qwen3): `modeling_qwen3.py:138 Qwen3RotaryEmbedding.forward` (4 op events): `inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)`
- synthetic qwen3 (test_prec_diag_hf_oracle_qwen3): `modeling_qwen3.py:139 Qwen3RotaryEmbedding.forward` (2 op events): `position_ids_expanded = position_ids[:, None, :].float()`
- synthetic qwen3 (test_prec_diag_hf_oracle_qwen3): `modeling_qwen3.py:143 Qwen3RotaryEmbedding.forward` (14 op events): `freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)`
- synthetic qwen3 (test_prec_diag_hf_oracle_qwen3): `modeling_qwen3.py:144 Qwen3RotaryEmbedding.forward` (2 op events): `emb = torch.cat((freqs, freqs), dim=-1)`
- synthetic qwen3 (test_prec_diag_hf_oracle_qwen3): `modeling_qwen3.py:145 Qwen3RotaryEmbedding.forward` (4 op events): `cos = emb.cos() * self.attention_scaling`
- synthetic qwen3 (test_prec_diag_hf_oracle_qwen3): `modeling_qwen3.py:146 Qwen3RotaryEmbedding.forward` (4 op events): `sin = emb.sin() * self.attention_scaling`
- synthetic qwen3 (test_prec_diag_hf_oracle_qwen3): `modeling_qwen3.py:148 Qwen3RotaryEmbedding.forward` (2 op events): `return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)`
- synthetic qwen3 (test_prec_diag_hf_oracle_qwen3): `modeling_qwen3.py:213 eager_attention_forward` (15 op events): `attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)`
- synthetic qwen3 (test_prec_diag_hf_oracle_qwen3): `modeling_qwen3.py:61 Qwen3RMSNorm.forward` (26 op events): `hidden_states = hidden_states.to(torch.float32)`
- synthetic qwen3 (test_prec_diag_hf_oracle_qwen3): `modeling_qwen3.py:62 Qwen3RMSNorm.forward` (52 op events): `variance = hidden_states.pow(2).mean(-1, keepdim=True)`
- synthetic qwen3 (test_prec_diag_hf_oracle_qwen3): `modeling_qwen3.py:63 Qwen3RMSNorm.forward` (78 op events): `hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)`
- synthetic qwen3 (test_prec_diag_hf_oracle_qwen3): `modeling_qwen3.py:64 Qwen3RMSNorm.forward` (13 op events): `return self.weight * hidden_states.to(input_dtype)`
- google/gemma-3-270m-it: `modeling_gemma3.py:139 Gemma3RMSNorm._norm` (1090 op events): `return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)`
- google/gemma-3-270m-it: `modeling_gemma3.py:142 Gemma3RMSNorm.forward` (218 op events): `output = self._norm(x.float())`
- google/gemma-3-270m-it: `modeling_gemma3.py:145 Gemma3RMSNorm.forward` (654 op events): `output = output * (1.0 + self.weight.float())`
- google/gemma-3-270m-it: `modeling_gemma3.py:146 Gemma3RMSNorm.forward` (109 op events): `return output.type_as(x)`
- google/gemma-3-270m-it: `modeling_gemma3.py:220 Gemma3RotaryEmbedding.forward` (72 op events): `inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)`
- google/gemma-3-270m-it: `modeling_gemma3.py:221 Gemma3RotaryEmbedding.forward` (36 op events): `position_ids_expanded = position_ids[:, None, :].float()`
- google/gemma-3-270m-it: `modeling_gemma3.py:225 Gemma3RotaryEmbedding.forward` (252 op events): `freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)`
- google/gemma-3-270m-it: `modeling_gemma3.py:226 Gemma3RotaryEmbedding.forward` (36 op events): `emb = torch.cat((freqs, freqs), dim=-1)`
- google/gemma-3-270m-it: `modeling_gemma3.py:227 Gemma3RotaryEmbedding.forward` (72 op events): `cos = emb.cos() * attention_scaling`
- google/gemma-3-270m-it: `modeling_gemma3.py:228 Gemma3RotaryEmbedding.forward` (72 op events): `sin = emb.sin() * attention_scaling`
- google/gemma-3-270m-it: `modeling_gemma3.py:230 Gemma3RotaryEmbedding.forward` (36 op events): `return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)`
- google/gemma-3-270m-it: `modeling_gemma3.py:305 eager_attention_forward` (90 op events): `attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)`
- Qwen/Qwen3-0.6B: `modeling_qwen3.py:138 Qwen3RotaryEmbedding.forward` (4 op events): `inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)`
- Qwen/Qwen3-0.6B: `modeling_qwen3.py:139 Qwen3RotaryEmbedding.forward` (2 op events): `position_ids_expanded = position_ids[:, None, :].float()`
- Qwen/Qwen3-0.6B: `modeling_qwen3.py:143 Qwen3RotaryEmbedding.forward` (14 op events): `freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)`
- Qwen/Qwen3-0.6B: `modeling_qwen3.py:144 Qwen3RotaryEmbedding.forward` (2 op events): `emb = torch.cat((freqs, freqs), dim=-1)`
- Qwen/Qwen3-0.6B: `modeling_qwen3.py:145 Qwen3RotaryEmbedding.forward` (4 op events): `cos = emb.cos() * self.attention_scaling`
- Qwen/Qwen3-0.6B: `modeling_qwen3.py:146 Qwen3RotaryEmbedding.forward` (4 op events): `sin = emb.sin() * self.attention_scaling`
- Qwen/Qwen3-0.6B: `modeling_qwen3.py:148 Qwen3RotaryEmbedding.forward` (2 op events): `return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)`
- Qwen/Qwen3-0.6B: `modeling_qwen3.py:213 eager_attention_forward` (140 op events): `attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)`
- Qwen/Qwen3-0.6B: `modeling_qwen3.py:61 Qwen3RMSNorm.forward` (226 op events): `hidden_states = hidden_states.to(torch.float32)`
- Qwen/Qwen3-0.6B: `modeling_qwen3.py:62 Qwen3RMSNorm.forward` (452 op events): `variance = hidden_states.pow(2).mean(-1, keepdim=True)`
- Qwen/Qwen3-0.6B: `modeling_qwen3.py:63 Qwen3RMSNorm.forward` (678 op events): `hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)`
- Qwen/Qwen3-0.6B: `modeling_qwen3.py:64 Qwen3RMSNorm.forward` (113 op events): `return self.weight * hidden_states.to(input_dtype)`

## What the dispatch level sees, and what it cannot

Sees:

- every ATen op dispatched during the A64 forward, after autograd and CompositeImplicitAutograd decomposition (F.linear appears as t + mm, a matmul as mm/bmm, softmax(dtype=) as _to_copy + _softmax)
- the dtype of every tensor each op reads, returns or writes (in-place and out= targets, from the op schema), and every floating dtype= argument
- every dtype conversion, with the Python call site that issued it, so a cast that is undone later still appears at both ends
- tensors created inside the forward in the default dtype (float32), e.g. torch.tensor(x) or torch.zeros(n)

Does not see:

- arithmetic inside one kernel (accumulators, internal temporaries of mm, mean, _softmax, cumsum, cumprod): measured instead by the kernel probes, after the forward, at every distinct shape and stride the forward used, on the same backend and threads
- elementwise kernels at shapes other than the probe vector: elementwise probes run once per scalar variant on a fixed-length contiguous vector
- computation outside torch (numpy, Python math): in the oracle these are the patched functions' float64 numpy witnesses and config scalars
- tensors created before the trace starts (parameters, buffers): audit part 1 checks every floating parameter and buffer is float64, and every parameter against the checkpoint
- a float64 tensor holding a value already rounded to float32 (e.g. a buffer computed in float32 at construction and widened later): dtype is not precision. Not observed under A64's load: from_pretrained(dtype=float64) builds buffers in float64, and the backfill records Gemma 3's embed_scale as sqrt(hidden) in float64
- state carried between forwards (KV or recurrent caches across decode steps): A64 is one prefill with use_cache=False; within it, every state tensor is produced and written by traced ops
- work on another Python thread: a dispatch mode is thread-local. The part fails a trace with fewer mm/addmm than the model has nn.Linear modules (attention's bmm does not count), which catches a forward, or any nn.Linear, run elsewhere; non-linear work (a norm, an activation) moved to another thread is still not seen

## Subjects

- synthetic gemma3_text (test_prec_diag_hf_oracle): family `gemma3_text`, modeling sha256 `221f147ccc0b16d7fa0aa007231ee0a6bd40fb37feb79eb9e537290498468aef`, ids `[2, 16, 53, 90, 127, 164, 201, 238, 275, 22, 59, 96, 133, 170, 207, 244, 281, 28, 65, 102, 139, 176, 213, 250]`
- synthetic qwen3 (test_prec_diag_hf_oracle_qwen3): family `qwen3`, modeling sha256 `fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab`, ids `[16, 53, 90, 127, 164, 201, 238, 275, 22, 59, 96, 133, 170, 207, 244, 281, 28, 65, 102, 139, 176, 213, 250, 287]`
- google/gemma-3-270m-it: family `gemma3_text`, modeling sha256 `221f147ccc0b16d7fa0aa007231ee0a6bd40fb37feb79eb9e537290498468aef`, revision `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`, ids `[2, 818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 3541, 506, 8858, 4856, 236761]`
- Qwen/Qwen3-0.6B: family `qwen3`, modeling sha256 `fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, ids `[785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 3143, 279, 14796, 6073, 13]`
