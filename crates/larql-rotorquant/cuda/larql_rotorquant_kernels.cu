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
