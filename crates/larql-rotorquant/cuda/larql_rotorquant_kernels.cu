#include "upstream/cpy-planar-iso.cu"

extern "C" void larql_rotorquant_cpy_f16_planar3(
    const void * src,
    void * dst,
    long long ne,
    cudaStream_t stream
) {
    ggml_cuda_cpy_f16_planar3((const char *) src, (char *) dst, ne, stream);
}

extern "C" void larql_rotorquant_cpy_f16_planar4(
    const void * src,
    void * dst,
    long long ne,
    cudaStream_t stream
) {
    ggml_cuda_cpy_f16_planar4((const char *) src, (char *) dst, ne, stream);
}

extern "C" void larql_rotorquant_cpy_f16_iso3(
    const void * src,
    void * dst,
    long long ne,
    cudaStream_t stream
) {
    ggml_cuda_cpy_f16_iso3((const char *) src, (char *) dst, ne, stream);
}

extern "C" void larql_rotorquant_cpy_f16_iso4(
    const void * src,
    void * dst,
    long long ne,
    cudaStream_t stream
) {
    ggml_cuda_cpy_f16_iso4((const char *) src, (char *) dst, ne, stream);
}

__device__ __forceinline__ float larql_rotorquant_dequant_3bit(uint8_t q, uint8_t signs, int j) {
    const uint8_t low = (q >> ((j & 3) * 2)) & 0x3;
    const uint8_t sign = (signs >> (j & 7)) & 0x1;
    return d_centroids_3bit[low | (sign << 2)];
}

__device__ __forceinline__ float larql_rotorquant_dequant_4bit(uint8_t q, int j) {
    return d_centroids_4bit[(q >> ((j & 1) * 4)) & 0xF];
}

__global__ void kernel_dequantize_planar3_f32(
    const block_planar3_0 * __restrict__ src,
    float * __restrict__ dst,
    int64_t n_blocks
) {
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const block_planar3_0 * blk = &src[ib];
    float * out = dst + ib * QK_PLANAR3;
    const float norm = __half2float(blk->norm);

    for (int p = 0; p < 64; p++) {
        const int j0 = p * 2;
        const int j1 = j0 + 1;
        const float r0 = larql_rotorquant_dequant_3bit(blk->qs[j0 / 4], blk->signs[j0 / 8], j0);
        const float r1 = larql_rotorquant_dequant_3bit(blk->qs[j1 / 4], blk->signs[j1 / 8], j1);
        const float c = d_planar_cos[p];
        const float s = d_planar_sin[p];
        out[j0] = (c * r0 + s * r1) * norm;
        out[j1] = (-s * r0 + c * r1) * norm;
    }
}

__global__ void kernel_dequantize_planar4_f32(
    const block_planar4_0 * __restrict__ src,
    float * __restrict__ dst,
    int64_t n_blocks
) {
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const block_planar4_0 * blk = &src[ib];
    float * out = dst + ib * QK_PLANAR4;
    const float norm = __half2float(blk->norm);

    for (int p = 0; p < 64; p++) {
        const int j0 = p * 2;
        const int j1 = j0 + 1;
        const float r0 = larql_rotorquant_dequant_4bit(blk->qs[j0 / 2], j0);
        const float r1 = larql_rotorquant_dequant_4bit(blk->qs[j1 / 2], j1);
        const float c = d_planar_cos[p];
        const float s = d_planar_sin[p];
        out[j0] = (c * r0 + s * r1) * norm;
        out[j1] = (-s * r0 + c * r1) * norm;
    }
}

__device__ __forceinline__ void larql_rotorquant_iso_inverse(
    float r0,
    float r1,
    float r2,
    float r3,
    int g,
    float norm,
    float * out
) {
    const float qw = d_iso_qw[g];
    const float qx = d_iso_qx[g];
    const float qy = d_iso_qy[g];
    const float qz = d_iso_qz[g];
    out[0] = ( qw * r0 + qx * r1 + qy * r2 + qz * r3) * norm;
    out[1] = (-qx * r0 + qw * r1 + qz * r2 - qy * r3) * norm;
    out[2] = (-qy * r0 - qz * r1 + qw * r2 + qx * r3) * norm;
    out[3] = (-qz * r0 + qy * r1 - qx * r2 + qw * r3) * norm;
}

__global__ void kernel_dequantize_iso3_f32(
    const block_iso3_0 * __restrict__ src,
    float * __restrict__ dst,
    int64_t n_blocks
) {
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const block_iso3_0 * blk = &src[ib];
    float * out = dst + ib * QK_ISO3;
    const float norm = __half2float(blk->norm);

    for (int g = 0; g < 32; g++) {
        const int j = g * 4;
        const float r0 = larql_rotorquant_dequant_3bit(blk->qs[j / 4], blk->signs[j / 8], j);
        const float r1 = larql_rotorquant_dequant_3bit(blk->qs[(j + 1) / 4], blk->signs[(j + 1) / 8], j + 1);
        const float r2 = larql_rotorquant_dequant_3bit(blk->qs[(j + 2) / 4], blk->signs[(j + 2) / 8], j + 2);
        const float r3 = larql_rotorquant_dequant_3bit(blk->qs[(j + 3) / 4], blk->signs[(j + 3) / 8], j + 3);
        larql_rotorquant_iso_inverse(r0, r1, r2, r3, g, norm, out + j);
    }
}

__global__ void kernel_dequantize_iso4_f32(
    const block_iso4_0 * __restrict__ src,
    float * __restrict__ dst,
    int64_t n_blocks
) {
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const block_iso4_0 * blk = &src[ib];
    float * out = dst + ib * QK_ISO4;
    const float norm = __half2float(blk->norm);

    for (int g = 0; g < 32; g++) {
        const int j = g * 4;
        const float r0 = larql_rotorquant_dequant_4bit(blk->qs[j / 2], j);
        const float r1 = larql_rotorquant_dequant_4bit(blk->qs[(j + 1) / 2], j + 1);
        const float r2 = larql_rotorquant_dequant_4bit(blk->qs[(j + 2) / 2], j + 2);
        const float r3 = larql_rotorquant_dequant_4bit(blk->qs[(j + 3) / 2], j + 3);
        larql_rotorquant_iso_inverse(r0, r1, r2, r3, g, norm, out + j);
    }
}

extern "C" void larql_rotorquant_dequantize_planar3_f32(
    const void * src,
    void * dst,
    long long ne,
    cudaStream_t stream
) {
    ggml_cuda_init_planar_iso_constants();
    const int64_t n_blocks = ne / QK_PLANAR3;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_dequantize_planar3_f32<<<blocks, threads, 0, stream>>>(
        (const block_planar3_0 *)src, (float *)dst, n_blocks);
}

extern "C" void larql_rotorquant_dequantize_planar4_f32(
    const void * src,
    void * dst,
    long long ne,
    cudaStream_t stream
) {
    ggml_cuda_init_planar_iso_constants();
    const int64_t n_blocks = ne / QK_PLANAR4;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_dequantize_planar4_f32<<<blocks, threads, 0, stream>>>(
        (const block_planar4_0 *)src, (float *)dst, n_blocks);
}

extern "C" void larql_rotorquant_dequantize_iso3_f32(
    const void * src,
    void * dst,
    long long ne,
    cudaStream_t stream
) {
    ggml_cuda_init_planar_iso_constants();
    const int64_t n_blocks = ne / QK_ISO3;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_dequantize_iso3_f32<<<blocks, threads, 0, stream>>>(
        (const block_iso3_0 *)src, (float *)dst, n_blocks);
}

extern "C" void larql_rotorquant_dequantize_iso4_f32(
    const void * src,
    void * dst,
    long long ne,
    cudaStream_t stream
) {
    ggml_cuda_init_planar_iso_constants();
    const int64_t n_blocks = ne / QK_ISO4;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_dequantize_iso4_f32<<<blocks, threads, 0, stream>>>(
        (const block_iso4_0 *)src, (float *)dst, n_blocks);
}

__device__ __forceinline__ uint8_t larql_rotorquant_unpack_lsb_code(
    const uint8_t * codes,
    int64_t bit_pos,
    uint8_t bits
) {
    const uint32_t mask = (1u << bits) - 1u;
    const int64_t byte = bit_pos >> 3;
    const uint32_t shift = (uint32_t)(bit_pos & 7);
    const uint32_t lo = codes[byte];
    const uint32_t hi = (shift + bits > 8) ? codes[byte + 1] : 0u;
    return (uint8_t)(((lo | (hi << 8)) >> shift) & mask);
}

__device__ __forceinline__ float larql_rotorquant_lm3_cpu_ref(uint8_t code) {
    switch (code & 0x7) {
        case 0: return -0.875f;
        case 1: return -0.625f;
        case 2: return -0.375f;
        case 3: return -0.125f;
        case 4: return  0.125f;
        case 5: return  0.375f;
        case 6: return  0.625f;
        default: return 0.875f;
    }
}

__device__ __forceinline__ void larql_rotorquant_cpu_ref_iso_coeffs(
    int rot,
    float * a,
    float * b,
    float * c
) {
    const float half = (float)rot * 1.5707963267948966f / 15.0f;
    float s;
    float w;
    sincosf(half, &s, &w);
    const float ss_over_3 = (s * s) / 3.0f;
    const float ws_over_sqrt3 = w * s * 0.5773502691896258f;
    *a = 1.0f - 4.0f * ss_over_3;
    *b = 2.0f * (ss_over_3 - ws_over_sqrt3);
    *c = 2.0f * (ss_over_3 + ws_over_sqrt3);
}

__global__ void kernel_dequantize_cpu_ref_iso3_f32(
    const uint8_t * __restrict__ codes,
    const float * __restrict__ norms,
    const uint16_t * __restrict__ rotation_indices,
    float * __restrict__ dst,
    int64_t n_rows,
    int64_t head_dim
) {
    const int64_t elem = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = n_rows * head_dim;
    if (elem >= total) return;

    const int64_t row = elem / head_dim;
    const int64_t col = elem - row * head_dim;
    const int64_t block = col / 4;
    const int lane = (int)(col - block * 4);
    const int64_t blocks_per_row = head_dim / 4;
    const int rot = (int)rotation_indices[row * blocks_per_row + block];
    const int64_t base_code = row * head_dim + block * 4;

    const float r0 = larql_rotorquant_lm3_cpu_ref(
        larql_rotorquant_unpack_lsb_code(codes, (base_code + 0) * 3, 3));
    const float r1 = larql_rotorquant_lm3_cpu_ref(
        larql_rotorquant_unpack_lsb_code(codes, (base_code + 1) * 3, 3));
    const float r2 = larql_rotorquant_lm3_cpu_ref(
        larql_rotorquant_unpack_lsb_code(codes, (base_code + 2) * 3, 3));
    const float r3 = larql_rotorquant_lm3_cpu_ref(
        larql_rotorquant_unpack_lsb_code(codes, (base_code + 3) * 3, 3));

    float a;
    float b;
    float c;
    larql_rotorquant_cpu_ref_iso_coeffs(rot, &a, &b, &c);
    float recovered;
    switch (lane) {
        case 0: recovered = a * r0 + c * r1 + b * r2; break;
        case 1: recovered = b * r0 + a * r1 + c * r2; break;
        case 2: recovered = c * r0 + b * r1 + a * r2; break;
        default: recovered = r3; break;
    }

    dst[elem] = recovered * norms[row];
}

extern "C" void larql_rotorquant_dequantize_cpu_ref_iso3_f32(
    const void * codes,
    const void * norms,
    const void * rotation_indices,
    void * dst,
    long long n_rows,
    long long head_dim,
    cudaStream_t stream
) {
    const int64_t total = n_rows * head_dim;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    kernel_dequantize_cpu_ref_iso3_f32<<<blocks, threads, 0, stream>>>(
        (const uint8_t *)codes,
        (const float *)norms,
        (const uint16_t *)rotation_indices,
        (float *)dst,
        n_rows,
        head_dim);
}
