# Quantization Formats

`larql-models` handles data format encoding/decoding. Compute-path quantized operations (GPU matvec, shader dispatch) are in `larql-compute`.

## f16 / bf16 (half.rs)

IEEE 754 half-precision and Google Brain bfloat16.

### f16 (binary16)

```
Sign: 1 bit | Exponent: 5 bits | Mantissa: 10 bits
Range: ±65504, precision: ~3.3 decimal digits
```

Conversion: bit manipulation (no hardware f16 required).

### bf16 (bfloat16)

```
Sign: 1 bit | Exponent: 8 bits | Mantissa: 7 bits
Range: same as f32, precision: ~2.4 decimal digits
```

Decode: shift left 16 bits → reinterpret as f32. Fast but less precise than f16.

### API

```rust
use larql_models::quant::half;

let f16_bytes = half::encode_f16(&[1.0, -2.0, 3.14]);  // Vec<u8>, 2 bytes each
let f32_vals = half::decode_f16(&f16_bytes);             // Vec<f32>
let f32_vals = half::decode_bf16(&bf16_bytes);           // Vec<f32>
```

## GGML Block Quantization (`quant/ggml/`)

GGML uses block quantization: groups of 32 or 256 elements share scale metadata,
reducing storage while preserving relative magnitudes within each block. The
loader currently dequantizes supported GGUF tensor types to f32 `ModelWeights`;
the fused row operations documented here are available for compute-side use.

### Q4_0

```
Block size: 32 elements
Storage: 2 bytes (f16 scale) + 16 bytes (32 × 4-bit values) = 18 bytes
Bits per weight: 4.5
```

Encoding: scale = max(abs(block)) / 7, quantize each value to 4-bit signed int.
Decoding: value = scale × (nibble - 8).

### Q4_1

```
Block size: 32 elements
Storage: 2 bytes (f16 scale) + 2 bytes (f16 min) + 16 bytes (4-bit values) = 20 bytes
Bits per weight: 5.0
```

Like Q4_0 but with a per-block minimum offset. Decoding: value = scale × nibble + min.

### Q5_0

```
Block size: 32 elements
Storage: 2 bytes (f16 scale) + 4 bytes (32 high bits) + 16 bytes (4-bit low values) = 22 bytes
Bits per weight: 5.5
```

5-bit quantization: 4 low bits packed as Q4_0, plus one extra bit per element stored in a separate bitfield.

### Q5_1

```
Block size: 32 elements
Storage: 2 bytes (f16 scale) + 2 bytes (f16 min) + 4 bytes (high bits) + 16 bytes (low values) = 24 bytes
Bits per weight: 6.0
```

Q5_0 with per-block minimum offset.

### Q8_0

```
Block size: 32 elements
Storage: 2 bytes (f16 scale) + 32 bytes (32 × int8 values) = 34 bytes
Bits per weight: 8.5
```

Encoding: scale = max(abs(block)) / 127, quantize to int8.
Decoding: value = scale × int8_value.

Higher quality than Q4 but 2x larger. Used for intermediate quantization in compute paths.

### Q4_K

```
Super-block size: 256 elements
Storage: 2 bytes (f16 d) + 2 bytes (f16 dmin) + 12 bytes (8 packed 6-bit scales+mins) + 128 bytes (nibbles) = 144 bytes
Bits per weight: 4.5
```

8 sub-blocks of 32 elements each. Each sub-block has its own 6-bit scale and min derived from the 12-byte packed field. Used for gate/up projections in Q4_K_M GGUF mixes.

### Q5_K

```
Super-block size: 256 elements
Storage: 2 bytes (f16 d) + 2 bytes (f16 dmin) + 12 bytes (8 packed 6-bit scales+mins)
         + 32 bytes (1 high bit per element) + 128 bytes (nibbles) = 176 bytes
Bits per weight: 5.5
```

Q4_K's affine layout widened by one bit. Identical `d` / `dmin` globals and
identical 12-byte 6-bit (scale, min) packing; the addition is a 32-byte `qh`
plane inserted before the nibbles, supplying a fifth magnitude bit per
element. Decode is `x = (d * q_scale) * q - (dmin * q_min)` with `q ∈ [0, 31]`
rather than Q4_K's `[0, 15]`. Element ordering matches Q4_K: group `g` covers
`qs[g*32 .. (g+1)*32]`, whose low nibbles feed sub-block `2g` and high nibbles
sub-block `2g+1`; `qh` bits `2g` / `2g+1` are those two halves' fifth bits.

Used for gate/up (and often down) projections in `Q5_K_M` GGUF mixes.

**CPU-only.** There is no Q5_K Metal shader. `QuantFormat::has_metal_kernel()`
reports `false` for it and the Metal decode paths refuse it up front
(`decode::preflight`) — being a 256-element k-quant, it would otherwise take
the Q4_K/Q6_K route and be read at a 144- or 210-byte stride instead of 176.

### Q6_K

```
Super-block size: 256 elements
Storage: 128 bytes (lower 4 bits) + 64 bytes (upper 2 bits) + 16 bytes (int8 scales) + 2 bytes (f16 d) = 210 bytes
Bits per weight: 6.5625
```

6-bit signed quantization with int8 per-16-element scales. Highest precision K-quant; used for down projections in Q4_K_M.

### K-quant API

```rust
use larql_models::quant::ggml::{q4_k, q5_k, q6_k};

// Fused decode + dot (no intermediate Vec allocation)
let dot: f32 = q4_k::q4k_row_dot(&row_bytes, &x)?;
let dot: f32 = q5_k::q5k_row_dot(&row_bytes, &x)?;
let dot: f32 = q6_k::q6k_row_dot(&row_bytes, &x)?;

// Fused decode + scaled-add: out += alpha * dequant(row)
q4_k::q4k_row_scaled_add(&row_bytes, alpha, &mut out)?;
q5_k::q5k_row_scaled_add(&row_bytes, alpha, &mut out)?;
q6_k::q6k_row_scaled_add(&row_bytes, alpha, &mut out)?;

// Full dequantize to Vec<f32>
let vals = q4_k::dequantize_q4_k(&bytes, num_elements)?;
let vals = q5_k::dequantize_q5_k(&bytes, num_elements)?;
let vals = q6_k::dequantize_q6_k(&bytes, num_elements)?;
```

On aarch64, `q4k_row_dot`, `q5k_row_dot` and `q6k_row_dot` use NEON SIMD;
other targets fall back to scalar. Tests assert NEON and scalar parity, plus
fused row-dot and scaled-add agreement with full dequantization.

Encoders (f32 → packed bytes) for the K-quants live in
`larql_compute::cpu::ops::q4_common` per ADR-008 — `quantize_q4_k`,
`quantize_q5_k`, `quantize_q6_k` — next to the SIMD kernels that consume them.

### API

```rust
use larql_models::quant::ggml;

// Quantize
let q4_bytes = ggml::quantize_q4_0(&f32_data);      // f32 → packed Q4_0
let q8_bytes = ggml::quantize_q8_0(&f32_data);      // f32 → packed Q8_0

// Dequantize (any supported type). Returns ModelError::Parse if `bytes`
// is shorter than the declared element count implies, or if num_elements
// is not a multiple of the block size (32 for Q4/Q5/Q8, 256 for Q4_K/Q6_K).
let f32_data = ggml::dequantize(&bytes, ggml::TYPE_Q4_0, num_elements)?;
let f32_data = ggml::dequantize_q4_0(&bytes, num_elements)?;  // type-specific

// Format info
let size = ggml::tensor_data_size(ggml::TYPE_Q4_K, 1024);  // bytes for 1024 elements
let name = ggml::type_name(ggml::TYPE_Q6_K);                // "Q6_K"
```

### Type Constants

| Constant | Value | Name |
|----------|-------|------|
| `TYPE_F32` | 0 | F32 |
| `TYPE_F16` | 1 | F16 |
| `TYPE_Q4_0` | 2 | Q4_0 |
| `TYPE_Q4_1` | 3 | Q4_1 |
| `TYPE_Q8_0` | 6 | Q8_0 |
| `TYPE_Q5_0` | 8 | Q5_0 |
| `TYPE_Q5_1` | 9 | Q5_1 |
| `TYPE_Q3_K` | 11 | Q3_K |
| `TYPE_Q4_K` | 12 | Q4_K |
| `TYPE_Q5_K` | 13 | Q5_K |
| `TYPE_Q6_K` | 14 | Q6_K |
| `TYPE_BF16` | 30 | BF16 |

`TYPE_Q2_K` is recognized for diagnostics and sizing compatibility but is not
dequantized; dispatch returns `ModelError::UnsupportedDtype` for unsupported
GGML types. `TYPE_Q3_K` and `TYPE_Q5_K` **are** dequantized — Q5_K also has
fused row ops and a registry entry, Q3_K currently has neither (decode only).

## MXFP4 (mxfp4.rs)

Microscaling FP4 format used by GPT-OSS / OpenAI models for packed MoE expert weights.

### Format

```
Block size: 32 elements
Scale: 1 byte e8m0 (8-bit exponent, no mantissa — pure power of 2)
Values: 32 × 4-bit floats (16 bytes), packed 2 per byte

Total: 1 + 16 = 17 bytes per 32 elements = 4.25 bits per weight
```

### e8m0 Scale

The scale is a pure power of 2 encoded as an 8-bit unsigned exponent:

```
scale = 2^(exponent - 127)

Examples:
  exponent=0   → 2^(-127) ≈ 5.88e-39 (denorm)
  exponent=126 → 2^(-1) = 0.5
  exponent=127 → 2^0 = 1.0
  exponent=128 → 2^1 = 2.0
  exponent=255 → 2^128 ≈ 3.4e38
```

### FP4 Values

Each 4-bit value encodes a small float: 1 sign bit + 2 exponent bits + 1 mantissa bit.

| Nibble | Value | Nibble | Value |
|--------|-------|--------|-------|
| 0000 | 0.0 | 1000 | -0.0 |
| 0001 | 0.5 | 1001 | -0.5 |
| 0010 | 1.0 | 1010 | -1.0 |
| 0011 | 1.5 | 1011 | -1.5 |
| 0100 | 2.0 | 1100 | -2.0 |
| 0101 | 3.0 | 1101 | -3.0 |
| 0110 | 4.0 | 1110 | -4.0 |
| 0111 | 6.0 | 1111 | -6.0 |

Final value = e8m0_scale × fp4_value.

### API

```rust
use larql_models::quant::mxfp4;

let scale = mxfp4::e8m0_to_f32(128);  // 2.0

// Dequantize a single expert's projection:
//   blocks = out_features × groups × 16 bytes (each byte = 2 × 4-bit values)
//   scales = out_features × groups bytes (one e8m0 per group of 32 elements)
let f32_row = mxfp4::dequantize_expert(&blocks, &scales, out_features, groups)?;

// Dequantize all experts from packed [num_experts, out_features, groups, 16] tensors:
let experts: Vec<Vec<f32>> =
    mxfp4::dequantize_all_experts(&blocks, &scales, num_experts, out_features, groups)?;

// Split GPT-OSS fused gate_up tensor into separate gate (w1) and up (w3) per-expert matrices.
// out_features = 2 × hidden (gate and up fused row-wise); splits at the midpoint.
let (gate_experts, up_experts): (ExpertWeights, ExpertWeights) =
    mxfp4::split_gate_up_experts(&blocks, &scales, num_experts, out_features, groups)?;
```

All functions return `ModelError::Parse` if `blocks` or `scales` is too short
for the declared shape — truncated inputs surface as clean errors rather than
panicking on a slice OOB.

## Size Comparison

For a 10240×2560 FFN weight matrix (26.2M elements):

| Format | Size | Ratio vs f32 |
|--------|------|-------------|
| f32 | 105 MB | 1.0x |
| f16 | 52.4 MB | 0.50x |
| Q8_0 | 27.9 MB | 0.27x |
| Q6_K | 21.4 MB | 0.20x |
| Q5_K | 18.0 MB | 0.17x |
| Q5_1 | 19.7 MB | 0.19x |
| Q5_0 | 18.0 MB | 0.17x |
| Q4_K | 14.6 MB | 0.14x |
| Q4_1 | 16.4 MB | 0.16x |
| Q4_0 | 14.7 MB | 0.14x |
| MXFP4 | 13.9 MB | 0.13x |
