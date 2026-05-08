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
