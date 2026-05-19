//! Per-layer GPU-buffer dump helpers used when
//! `LARQL_METAL_DUMP_LAYERS=<dir>` is set.
//!
//! Pulled out of `dispatch_full_pipeline` so the orchestrator's body
//! stays focused on compute, not on `eprintln`/IO. All functions
//! commit + wait on the supplied command buffer first (you can't read
//! GPU buffers mid-pipeline) and return a fresh command buffer to
//! continue the dispatch.

use metal::{Buffer, CommandBuffer, CommandQueue};

use super::buffers::LayerBuffers;
use larql_compute::FullPipelineLayer;

/// Read `n` f32s out of a Metal `Buffer` and write them as raw
/// little-endian bytes to `<dir>/<name>`.
fn write_f32_buffer(dir: &str, name: &str, buf: &Buffer, n: usize) {
    let ptr = buf.contents() as *const f32;
    if ptr.is_null() {
        return;
    }
    // SAFETY: Caller commits + waits before this is invoked, so the
    // buffer is finished writing on the GPU side. `n` is sized to the
    // buffer's logical row count and the buffer was allocated for at
    // least `n * 4` bytes.
    let s = unsafe { std::slice::from_raw_parts(ptr, n) };
    let bytes: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
    let path = format!("{dir}/{name}");
    if let Err(e) = std::fs::write(&path, &bytes) {
        eprintln!("[dump] failed to write {path}: {e}");
    }
}

/// Dump the input embedding (h_bufs[0]) before any layer compute runs.
/// Lets a CPU/Metal bisect verify both sides start from the same point.
pub(super) fn dump_h_embed(
    dump_dir: Option<&str>,
    lb: &LayerBuffers,
    seq_len: usize,
    hidden: usize,
) {
    let Some(dir) = dump_dir else {
        return;
    };
    write_f32_buffer(dir, "metal_h_embed.f32", &lb.h[0], seq_len * hidden);
}

/// One-off mid-pipeline dump of `q_out[0]` after a specific stage —
/// used to bisect whether QKV-projection or QK-norm is responsible for
/// drift. Commits + waits the supplied `cmd`, then re-issues a fresh
/// command buffer.
#[allow(clippy::too_many_arguments)]
pub(super) fn dump_layer0_q_after_stage(
    dump_dir: Option<&str>,
    queue: &CommandQueue,
    cmd: CommandBuffer,
    lb: &LayerBuffers,
    stage_name: &str,
    seq_len: usize,
    layer_q_dim: usize,
    layer_idx: usize,
) -> CommandBuffer {
    let Some(dir) = dump_dir else {
        return cmd;
    };
    if layer_idx != 0 {
        return cmd;
    }
    cmd.commit();
    cmd.wait_until_completed();
    let name = format!("metal_L0_q_out_{stage_name}.f32");
    write_f32_buffer(dir, &name, &lb.q_out[layer_idx], seq_len * layer_q_dim);
    queue.new_command_buffer().to_owned()
}

/// End-of-layer snapshot: writes `metal_layer_NN_<stage>.f32` for the
/// post-residual hidden state and the per-stage scratch buffers (the
/// latter only for the layer named by `LARQL_STAGE_DUMP_LAYER`).
/// Commits + waits the supplied `cmd`, then returns a fresh one.
#[allow(clippy::too_many_arguments)]
pub(super) fn dump_layer_snapshots(
    dump_dir: Option<&str>,
    queue: &CommandQueue,
    cmd: CommandBuffer,
    lb: &LayerBuffers,
    layers: &[FullPipelineLayer<'_>],
    l: usize,
    seq_len: usize,
    hidden: usize,
    inter: usize,
) -> CommandBuffer {
    let Some(dir) = dump_dir else {
        return cmd;
    };
    cmd.commit();
    cmd.wait_until_completed();
    let layer_q_dim = layers[l].num_q_heads * layers[l].head_dim;
    let layer_kv_dim = layers[l].num_kv_heads * layers[l].head_dim;
    let layer_dump = |name: &str, buf: &Buffer, n: usize| {
        write_f32_buffer(dir, &format!("metal_layer_{l:02}_{name}.f32"), buf, n);
    };

    // End-of-layer residual (matches CPU dump exactly).
    layer_dump("h_out", &lb.h[l + 1], seq_len * hidden);
    // h_post_attn for every layer — cheap and lets the residual-diff
    // tool bisect drift into attention vs FFN at any layer. Without
    // this, L0 was the only layer with this snapshot available.
    layer_dump("h_post_attn", &lb.h_post_attn[l], seq_len * hidden);
    // Per-stage snapshots for layer 0 by default, or the layer named
    // by `LARQL_STAGE_DUMP_LAYER` — useful for bisecting drift at a
    // specific later layer (e.g. Gemma 4 global L5).
    let stage_layer =
        larql_compute::options::env_usize(larql_compute::options::ENV_STAGE_DUMP_LAYER)
            .unwrap_or(0);
    if l == stage_layer {
        layer_dump("norm_out", &lb.norm_out[l], seq_len * hidden);
        layer_dump("q_out", &lb.q_out[l], seq_len * layer_q_dim);
        layer_dump("k_out", &lb.k_out[l], seq_len * layer_kv_dim);
        layer_dump("v_out", &lb.v_out[l], seq_len * layer_kv_dim);
        layer_dump("attn_out", &lb.attn_out[l], seq_len * layer_q_dim);
        layer_dump("o_out", &lb.o_out[l], seq_len * hidden);
        layer_dump("ffn_norm_out", &lb.ffn_norm_out[l], seq_len * hidden);
        layer_dump("gate_out", &lb.gate_out[l], seq_len * inter);
        layer_dump("up_out", &lb.up_out[l], seq_len * inter);
        layer_dump("act_buf", &lb.act_buf[l], seq_len * inter);
        layer_dump("down_out", &lb.down_out[l], seq_len * hidden);
    }
    queue.new_command_buffer().to_owned()
}
