//! f32 tiled matrix multiply transposed: C = A × B^T.

pub const SHADER: &str = r#"
constant uint TS_T = 32;

kernel void sgemm_transb(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    threadgroup float As[TS_T][TS_T];
    threadgroup float Bs[TS_T][TS_T];
    uint row = gid.y * TS_T + tid.y;
    uint col = gid.x * TS_T + tid.x;
    float acc = 0.0f;
    uint tiles = (K + TS_T - 1) / TS_T;
    for (uint t = 0; t < tiles; t++) {
        uint ac = t * TS_T + tid.x;
        uint bk = t * TS_T + tid.y;
        As[tid.y][tid.x] = (row < M && ac < K) ? A[row * K + ac] : 0.0f;
        Bs[tid.y][tid.x] = (col < N && bk < K) ? B[col * K + bk] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < TS_T; i++) acc = fma(As[tid.y][i], Bs[i][tid.x], acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "sgemm_transb";
}
