//! f32 tiled matrix multiply: C = A × B.
//! Tile size 32×32, threadgroup shared memory.

pub const SHADER: &str = r#"
constant uint TS = 32;

kernel void sgemm(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    threadgroup float As[TS][TS];
    threadgroup float Bs[TS][TS];
    uint row = gid.y * TS + tid.y;
    uint col = gid.x * TS + tid.x;
    float acc = 0.0f;
    uint tiles = (K + TS - 1) / TS;
    for (uint t = 0; t < tiles; t++) {
        uint ac = t * TS + tid.x;
        uint br = t * TS + tid.y;
        As[tid.y][tid.x] = (row < M && ac < K) ? A[row * K + ac] : 0.0f;
        Bs[tid.y][tid.x] = (br < K && col < N) ? B[br * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < TS; i++) acc = fma(As[tid.y][i], Bs[i][tid.x], acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "sgemm";
}
