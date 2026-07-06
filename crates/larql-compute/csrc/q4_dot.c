// Fused Q4_0 × Q8_0 dot product with ARM dotprod intrinsics.
// Compiled with clang -O3 -march=armv8.2-a+dotprod.
// Called from Rust via FFI — one function, no dependencies.

#include <stdint.h>
#include <stddef.h>
#include <string.h>

// Helper: decode f16 to f32
static inline float decode_f16(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            uint32_t result = sign;
            float f;
            memcpy(&f, &result, sizeof(f));
            return f;
        }
        exponent = 1;
        while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
        mantissa &= 0x3FF;
        exponent += 127 - 15;
    } else if (exponent == 31) {
        exponent = 255;
    } else {
        exponent += 127 - 15;
    }

    uint32_t result = sign | (exponent << 23) | (mantissa << 13);
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
}

#if defined(__aarch64__)
#include <arm_neon.h>

// Fused Q4_0 × Q8_0 dot product for one row.
//
// q4_row:    packed Q4_0 blocks (18 bytes each: 2B f16 scale + 16B nibbles)
// q8:        pre-quantized Q8_0 input vector (int8)
// q8_scales: per-block scales for the Q8 input (float32)
// blocks:    number of Q4_0 blocks (hidden_size / 32)
//
// Returns: dot product (float32)
float q4_q8_dot_neon_c(
    const uint8_t* q4_row,
    const int8_t* q8,
    const float* q8_scales,
    size_t blocks
) {
    float acc = 0.0f;
    const int8x16_t offset = vdupq_n_s8(8);
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);

    for (size_t b = 0; b < blocks; b++) {
        const uint8_t* block = q4_row + b * 18;

        // Decode f16 scale
        uint16_t scale_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
        float q4_scale = decode_f16(scale_bits);
        float combined_scale = q4_scale * q8_scales[b];

        const uint8_t* quants = block + 2;
        const int8_t* q8_ptr = q8 + b * 32;

        // Load 16 bytes of packed Q4 nibbles
        uint8x16_t raw = vld1q_u8(quants);

        // Split into low/high nibbles, subtract 8 for signed range.
        // ggml planar Q4_0 layout (quantize_row_q4_0_ref): low nibbles are
        // elements 0..16, high nibbles are elements 16..32 — they pair with
        // the sequential Q8 input directly, no interleave needed.
        int8x16_t q4_0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), offset);
        int8x16_t q4_1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), offset);

        // Load Q8 values
        int8x16_t q8_0 = vld1q_s8(q8_ptr);
        int8x16_t q8_1 = vld1q_s8(q8_ptr + 16);

#if defined(__ARM_FEATURE_DOTPROD)
        // vdotq_s32: 4 int8 pairs → 1 int32, per lane. 16 pairs per instruction.
        int32x4_t isum = vdupq_n_s32(0);
        isum = vdotq_s32(isum, q4_0, q8_0);
        isum = vdotq_s32(isum, q4_1, q8_1);
        acc += (float)vaddvq_s32(isum) * combined_scale;
#else
        // Fallback: vmull + pairwise add
        int16x8_t prod0 = vmull_s8(vget_low_s8(q4_0), vget_low_s8(q8_0));
        int16x8_t prod1 = vmull_s8(vget_high_s8(q4_0), vget_high_s8(q8_0));
        int16x8_t prod2 = vmull_s8(vget_low_s8(q4_1), vget_low_s8(q8_1));
        int16x8_t prod3 = vmull_s8(vget_high_s8(q4_1), vget_high_s8(q8_1));
        int16x8_t sum16 = vaddq_s16(vaddq_s16(prod0, prod1), vaddq_s16(prod2, prod3));
        int32x4_t sum32 = vpaddlq_s16(sum16);
        acc += (float)vaddvq_s32(sum32) * combined_scale;
#endif
    }

    return acc;
}

// Fused Q4_0 matvec: scores[num_rows] = Q4_matrix[num_rows, hidden] @ Q8_x[hidden]
// Calls q4_q8_dot_neon_c per row.
void q4_0_matvec_c(
    const uint8_t* q4_data,
    const int8_t* q8_x,
    const float* q8_scales,
    float* scores,
    size_t num_rows,
    size_t hidden
) {
    size_t blocks_per_row = hidden / 32;
    size_t bytes_per_row = blocks_per_row * 18;

    for (size_t row = 0; row < num_rows; row++) {
        scores[row] = q4_q8_dot_neon_c(
            q4_data + row * bytes_per_row,
            q8_x,
            q8_scales,
            blocks_per_row
        );
    }
}

// Fused Q4_0 vecmat: out[hidden] = activation[intermediate] @ Q4[intermediate, hidden]
// Scalar with zero-skip. NEON accelerates the inner dequant loop.
void q4_0_vecmat_c(
    const float* activation,
    const uint8_t* q4_data,
    float* out,
    size_t intermediate,
    size_t hidden
) {
    size_t blocks_per_row = hidden / 32;
    size_t bytes_per_row = blocks_per_row * 18;

    for (size_t j = 0; j < hidden; j++) out[j] = 0.0f;

    for (size_t row = 0; row < intermediate; row++) {
        float act = activation[row];
        if (act > -1e-10f && act < 1e-10f) continue;

        const uint8_t* row_data = q4_data + row * bytes_per_row;

        for (size_t b = 0; b < blocks_per_row; b++) {
            const uint8_t* block = row_data + b * 18;
            uint16_t scale_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
            float scale = decode_f16(scale_bits) * act;
            const uint8_t* quants = block + 2;
            float* o = out + b * 32;

            for (size_t j = 0; j < 16; j++) {
                uint8_t byte = quants[j];
                int lo_v = (byte & 0x0F) - 8;
                int hi_v = ((byte >> 4) & 0x0F) - 8;
                // ggml planar layout: lo → element j, hi → element j+16.
                o[j]      += (float)lo_v * scale;
                o[j + 16] += (float)hi_v * scale;
            }
        }
    }
}

#else
// Non-ARM scalar fallback

void q4_0_matvec_c(
    const uint8_t* q4_data,
    const int8_t* q8_x,
    const float* q8_scales,
    float* scores,
    size_t num_rows,
    size_t hidden
) {
    size_t blocks_per_row = hidden / 32;
    size_t bytes_per_row = blocks_per_row * 18;

    for (size_t row = 0; row < num_rows; row++) {
        float acc = 0.0f;
        const uint8_t* row_data = q4_data + row * bytes_per_row;
        for (size_t b = 0; b < blocks_per_row; b++) {
            const uint8_t* block = row_data + b * 18;
            uint16_t scale_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
            float combined_scale = decode_f16(scale_bits) * q8_scales[b];
            const uint8_t* quants = block + 2;
            const int8_t* q8_ptr = q8_x + b * 32;
            for (size_t j = 0; j < 16; j++) {
                uint8_t byte = quants[j];
                int lo_v = (byte & 0x0F) - 8;
                int hi_v = ((byte >> 4) & 0x0F) - 8;
                // ggml planar layout: lo → element j, hi → element j+16.
                acc += (float)lo_v * (float)q8_ptr[j]      * combined_scale;
                acc += (float)hi_v * (float)q8_ptr[j + 16] * combined_scale;
            }
        }
        scores[row] = acc;
    }
}

void q4_0_vecmat_c(
    const float* activation,
    const uint8_t* q4_data,
    float* out,
    size_t intermediate,
    size_t hidden
) {
    size_t blocks_per_row = hidden / 32;
    size_t bytes_per_row = blocks_per_row * 18;

    for (size_t j = 0; j < hidden; j++) out[j] = 0.0f;

    for (size_t row = 0; row < intermediate; row++) {
        float act = activation[row];
        if (act > -1e-10f && act < 1e-10f) continue;
        const uint8_t* row_data = q4_data + row * bytes_per_row;
        for (size_t b = 0; b < blocks_per_row; b++) {
            const uint8_t* block = row_data + b * 18;
            uint16_t scale_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
            float scale = decode_f16(scale_bits) * act;
            const uint8_t* quants = block + 2;
            float* o = out + b * 32;
            for (size_t j = 0; j < 16; j++) {
                uint8_t byte = quants[j];
                int lo_v = (byte & 0x0F) - 8;
                int hi_v = ((byte >> 4) & 0x0F) - 8;
                // ggml planar layout: lo → element j, hi → element j+16.
                o[j]      += (float)lo_v * scale;
                o[j + 16] += (float)hi_v * scale;
            }
        }
    }
}
#endif
