//! Sparse Q4 matvec: score only K rows selected by index.
//!
//! out[K] = Q4[indices[K], hidden] @ Q8_x[hidden]
//!
//! Instead of scanning all N rows, reads K rows by index array.
//! Used when template routing narrows the feature universe.
//! At K=400 out of N=10240, reads 96% less data.
//!
//! Each thread handles one selected row. No simd reduction needed.
//!
//! ## Retention rationale (ADR-017)
//!
//! **Status**: experimental, diag-only. Built into `all_shaders()`
//! for the bench harness; not wired into production. Designed for a
//! template-routed feature dispatch pattern (score K≪N rows) that
//! the production decode path does not currently take.
//!
//! Kept on disk because the template-routing dispatch the larql
//! research arc is exploring (see Tier 2 / variable-argument dispatch
//! memos) is the natural caller for a "score only K rows by index"
//! kernel; reviving requires one dispatch site, not a kernel rewrite.
//!
//! **Removal trigger**: if template-routed dispatch ships through a
//! different mechanism (CPU pre-selection + dense GPU matvec, or
//! row-permuted weights on disk), demote.

pub const SHADER: &str = r#"
kernel void q4_sparse_matvec(
    device const uchar*  Q4       [[buffer(0)]],   // [N, hidden] Q4 packed (full weight matrix)
    device const char*   Q8       [[buffer(1)]],   // [hidden] Q8 input
    device const float*  Q8s      [[buffer(2)]],   // [blocks] Q8 scales
    device const uint*   indices  [[buffer(3)]],   // [K] row indices to score
    device float*        out      [[buffer(4)]],   // [K] output scores
    constant uint&       K        [[buffer(5)]],   // number of selected rows
    constant uint&       hidden   [[buffer(6)]],   // hidden dimension
    uint tid [[thread_position_in_grid]])
{
    if (tid >= K) return;

    uint row_idx = indices[tid];
    uint blocks = hidden / 32;
    uint bytes_per_row = blocks * 18;
    device const uchar* row = Q4 + row_idx * bytes_per_row;

    float acc = 0.0f;
    for (uint b = 0; b < blocks; b++) {
        device const uchar* blk = row + b * 18;
        ushort sb = ushort(blk[0]) | (ushort(blk[1]) << 8);
        float cs = decode_f16_metal(sb) * Q8s[b];
        device const uchar* qb = blk + 2;
        device const char* q8 = Q8 + b * 32;

        uint w0 = uint(qb[0]) | (uint(qb[1]) << 8) | (uint(qb[2]) << 16) | (uint(qb[3]) << 24);
        uint w1 = uint(qb[4]) | (uint(qb[5]) << 8) | (uint(qb[6]) << 16) | (uint(qb[7]) << 24);
        uint w2 = uint(qb[8]) | (uint(qb[9]) << 8) | (uint(qb[10]) << 16) | (uint(qb[11]) << 24);
        uint w3 = uint(qb[12]) | (uint(qb[13]) << 8) | (uint(qb[14]) << 16) | (uint(qb[15]) << 24);

        int isum = 0;
        #define D8(w, o) \
            isum += (int((w>> 0)&0xFu)-8)*int(q8[o+0]) + (int((w>> 4)&0xFu)-8)*int(q8[o+1]) \
                  + (int((w>> 8)&0xFu)-8)*int(q8[o+2]) + (int((w>>12)&0xFu)-8)*int(q8[o+3]) \
                  + (int((w>>16)&0xFu)-8)*int(q8[o+4]) + (int((w>>20)&0xFu)-8)*int(q8[o+5]) \
                  + (int((w>>24)&0xFu)-8)*int(q8[o+6]) + (int((w>>28)&0xFu)-8)*int(q8[o+7]);
        D8(w0,0); D8(w1,8); D8(w2,16); D8(w3,24);
        #undef D8

        acc += float(isum) * cs;
    }

    out[tid] = acc;
}
"#;
