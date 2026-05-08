//! Debug: per-stage buffer reads in the decode pipeline.
//! Runs inside larql-compute where we have direct Metal access.
//!
//! cargo run --release --features metal -p larql-compute --example debug_decode_pipeline

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("This example requires --features metal");
}

#[cfg(feature = "metal")]
fn main() {
    let metal = larql_compute::metal::MetalBackend::new().expect("need metal");
    let bufs = metal.bufs();
    let queue = metal.queue();

    println!("=== Decode Pipeline Stage Debug ===\n");

    let hidden = 2560;
    let q_dim = 2048; // 8 heads × 256
    let kv_dim = 1024; // 4 heads × 256
    let head_dim = 256;
    let num_q = 8;
    let num_kv = 4;
    let inter = 10240;

    // Synthetic input (nonzero)
    let x: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32 - 1280.0) * 0.01).sin() * 10.0)
        .collect();
    let x_max = x.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    println!("Input: len={}, max={:.4}", x.len(), x_max);

    // Synthetic weights (small random via Q4_0 quantize/dequantize roundtrip)
    let dummy_norm: Vec<f32> = vec![1.0; hidden];
    let gate_f32: Vec<f32> = (0..inter * hidden)
        .map(|i| ((i as f32) * 0.000001).sin() * 0.1)
        .collect();
    let dummy_gate_q4 = larql_compute::cpu::ops::q4_common::quantize_q4_0(&gate_f32);
    let dummy_up_q4 = larql_compute::cpu::ops::q4_common::quantize_q4_0(&gate_f32);
    let down_f32: Vec<f32> = (0..hidden * inter)
        .map(|i| ((i as f32) * 0.000002).cos() * 0.1)
        .collect();
    let dummy_down_q4 = larql_compute::cpu::ops::q4_common::quantize_q4_0(&down_f32);

    // Build Q4_K weights for attention (synthetic)
    let wq_data = larql_compute::cpu::ops::q4_common::quantize_q4_k(
        &(0..q_dim * hidden)
            .map(|i| ((i as f32) * 0.00001).sin() * 0.5)
            .collect::<Vec<_>>(),
    );
    let wk_data = larql_compute::cpu::ops::q4_common::quantize_q4_k(
        &(0..kv_dim * hidden)
            .map(|i| ((i as f32) * 0.00002).cos() * 0.5)
            .collect::<Vec<_>>(),
    );
    let wv_data = larql_compute::cpu::ops::q4_common::quantize_q4_k(
        &(0..kv_dim * hidden)
            .map(|i| ((i as f32) * 0.00003).sin() * 0.5)
            .collect::<Vec<_>>(),
    );
    let wo_data = larql_compute::cpu::ops::q4_common::quantize_q4_k(
        &(0..hidden * q_dim)
            .map(|i| ((i as f32) * 0.00004).cos() * 0.5)
            .collect::<Vec<_>>(),
    );

    use larql_compute::{
        Activation, FfnType, FullPipelineLayer, NormType, QuantFormat, QuantWeight,
    };

    let layer = FullPipelineLayer {
        wq: QuantWeight {
            data: &wq_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wk: QuantWeight {
            data: &wk_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wv: QuantWeight {
            data: &wv_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wo: QuantWeight {
            data: &wo_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        gate: QuantWeight {
            data: &dummy_gate_q4,
            scales: None,
            format: QuantFormat::Q4_0,
        },
        up: QuantWeight {
            data: &dummy_up_q4,
            scales: None,
            format: QuantFormat::Q4_0,
        },
        down: QuantWeight {
            data: &dummy_down_q4,
            scales: None,
            format: QuantFormat::Q4_0,
        },
        input_norm: &dummy_norm,
        post_attn_norm: &dummy_norm,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        norm_offset: 0.0,
        has_post_norms: false,
        activation: Activation::Silu,
        qk_norm_offset: 0.0,
        eps: 1e-6,
        norm_type: NormType::RmsNorm,
        ffn_type: FfnType::Gated,
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        head_dim,
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        rope_base: 10000.0,
        rotary_dim: 0,
        sliding_window: 0,
        has_v_norm: false,
        layer_scalar: 0.0,
        input_norm_bias: None,
        post_attn_norm_bias: None,
        q_norm_weight: None,
        k_norm_weight: None,
        ffn_up_bias: None,
        ffn_down_bias: None,
        moe: None,
        ffn_is_remote: false,
        moe_combined_output_norm: false,
        moe_outer_post_norm: None,
    };

    // Test 1: All-Q4_K (synthetic, matching formats)
    println!("\n--- Test 1: All Q4_K (uniform format) ---");
    let mut kv = metal.create_kv_cache(1, 4096, num_kv, head_dim);
    let result = larql_compute::metal::MetalBackend::decode_token(
        &metal,
        &mut kv,
        &[layer],
        &x,
        hidden,
        inter,
        q_dim,
        kv_dim,
        num_q,
        num_kv,
        head_dim,
        10000.0,
    );
    let nz = result.iter().filter(|v| v.abs() > 1e-10).count();
    let max = result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    println!("  decode_token: nonzero={}/{}, max={:.4}", nz, hidden, max);

    // Test 2: Standalone rms_norm with offset=1.0
    println!("\n--- Test 2: Standalone rms_norm (offset=1.0) ---");
    {
        let h_buf = bufs.transient_from_f32(&x);
        let norm_w = bufs.transient_from_f32(&vec![1.0f32; hidden]);
        let norm_out = bufs.output((hidden * 4) as u64);
        let eps = 1e-6f32;
        let offset = 1.0f32;
        let hidden_val = hidden as u32;
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&metal.rms_norm_pipeline);
        enc.set_buffer(0, Some(&h_buf), 0);
        enc.set_buffer(1, Some(&norm_w), 0);
        enc.set_buffer(2, Some(&norm_out), 0);
        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
        enc.dispatch_threads(
            metal::MTLSize::new(hidden as u64, 1, 1),
            metal::MTLSize::new(256.min(hidden as u64), 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let result = larql_compute::metal::buffers::read_buffer_f32(&norm_out, hidden);
        let nz = result.iter().filter(|v| v.abs() > 1e-10).count();
        let max = result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!(
            "  rms_norm(offset=1.0): nonzero={}/{}, max={:.4}",
            nz, hidden, max
        );
    }

    // Test 3: residual_norm_q8 with offset=1.0
    println!("\n--- Test 3: residual_norm_q8 (offset=1.0) ---");
    {
        let a_buf = bufs.transient_from_f32(&x);
        let b_buf = bufs.transient_from_f32(&vec![0.0f32; hidden]); // zero residual
        let norm_w = bufs.transient_from_f32(&vec![1.0f32; hidden]);
        let q8_out = bufs.output(hidden as u64);
        let q8_scales = bufs.output((hidden / 32 * 4) as u64);
        let f32_out = bufs.output((hidden * 4) as u64);
        let hidden_val = hidden as u32;
        let eps = 1e-6f32;
        let offset = 1.0f32;

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&metal.residual_norm_q8_pipeline);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&b_buf), 0);
        enc.set_buffer(2, Some(&norm_w), 0);
        enc.set_buffer(3, Some(&q8_out), 0);
        enc.set_buffer(4, Some(&q8_scales), 0);
        enc.set_buffer(5, Some(&f32_out), 0);
        enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(8, 4, &offset as *const f32 as *const std::ffi::c_void);
        enc.dispatch_threads(
            metal::MTLSize::new(hidden as u64, 1, 1),
            metal::MTLSize::new(256.min(hidden as u64), 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let f32_result = larql_compute::metal::buffers::read_buffer_f32(&f32_out, hidden);
        let nz = f32_result.iter().filter(|v| v.abs() > 1e-10).count();
        println!("  f32_out (residual): nonzero={}/{}", nz, hidden);

        let q8_data = larql_compute::metal::buffers::read_buffer_f32(&q8_scales, hidden / 32);
        let nz_scales = q8_data.iter().filter(|v| v.abs() > 1e-10).count();
        println!("  Q8 scales: nonzero={}/{}", nz_scales, hidden / 32);
    }

    // Test 4: decode with offset=1.0 only (toggle)
    println!("\n--- Test 4: decode_token with norm_offset=1.0 ---");
    {
        let layer4 = FullPipelineLayer {
            wq: QuantWeight {
                data: &wq_data,
                scales: None,
                format: QuantFormat::Q4_K,
            },
            wk: QuantWeight {
                data: &wk_data,
                scales: None,
                format: QuantFormat::Q4_K,
            },
            wv: QuantWeight {
                data: &wv_data,
                scales: None,
                format: QuantFormat::Q4_K,
            },
            wo: QuantWeight {
                data: &wo_data,
                scales: None,
                format: QuantFormat::Q4_K,
            },
            gate: QuantWeight {
                data: &dummy_gate_q4,
                scales: None,
                format: QuantFormat::Q4_0,
            },
            up: QuantWeight {
                data: &dummy_up_q4,
                scales: None,
                format: QuantFormat::Q4_0,
            },
            down: QuantWeight {
                data: &dummy_down_q4,
                scales: None,
                format: QuantFormat::Q4_0,
            },
            input_norm: &dummy_norm,
            post_attn_norm: &dummy_norm,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            norm_offset: 1.0,
            has_post_norms: false,
            activation: Activation::Silu,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            head_dim,
            num_q_heads: num_q,
            num_kv_heads: num_kv,
            rope_base: 10000.0,
            rotary_dim: 0,
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
            moe: None,
            ffn_is_remote: false,
            moe_combined_output_norm: false,
            moe_outer_post_norm: None,
        };
        let mut kv4 = metal.create_kv_cache(1, 4096, num_kv, head_dim);
        let r = larql_compute::metal::MetalBackend::decode_token(
            &metal,
            &mut kv4,
            &[layer4],
            &x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q,
            num_kv,
            head_dim,
            10000.0,
        );
        let nz = r.iter().filter(|v| v.abs() > 1e-10).count();
        let max = r.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("  nonzero={}/{}, max={:.4}", nz, hidden, max);
    }

    // Test 5: decode with GeluTanh only
    println!("\n--- Test 5: decode_token with activation=GeluTanh ---");
    {
        let layer5 = FullPipelineLayer {
            wq: QuantWeight {
                data: &wq_data,
                scales: None,
                format: QuantFormat::Q4_K,
            },
            wk: QuantWeight {
                data: &wk_data,
                scales: None,
                format: QuantFormat::Q4_K,
            },
            wv: QuantWeight {
                data: &wv_data,
                scales: None,
                format: QuantFormat::Q4_K,
            },
            wo: QuantWeight {
                data: &wo_data,
                scales: None,
                format: QuantFormat::Q4_K,
            },
            gate: QuantWeight {
                data: &dummy_gate_q4,
                scales: None,
                format: QuantFormat::Q4_0,
            },
            up: QuantWeight {
                data: &dummy_up_q4,
                scales: None,
                format: QuantFormat::Q4_0,
            },
            down: QuantWeight {
                data: &dummy_down_q4,
                scales: None,
                format: QuantFormat::Q4_0,
            },
            input_norm: &dummy_norm,
            post_attn_norm: &dummy_norm,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            norm_offset: 0.0,
            has_post_norms: false,
            activation: Activation::GeluTanh,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            head_dim,
            num_q_heads: num_q,
            num_kv_heads: num_kv,
            rope_base: 10000.0,
            rotary_dim: 0,
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
            moe: None,
            ffn_is_remote: false,
            moe_combined_output_norm: false,
            moe_outer_post_norm: None,
        };
        let mut kv5 = metal.create_kv_cache(1, 4096, num_kv, head_dim);
        let r = larql_compute::metal::MetalBackend::decode_token(
            &metal,
            &mut kv5,
            &[layer5],
            &x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q,
            num_kv,
            head_dim,
            10000.0,
        );
        let nz = r.iter().filter(|v| v.abs() > 1e-10).count();
        let max = r.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("  nonzero={}/{}, max={:.4}", nz, hidden, max);
    }

    println!("\n=== Done ===");
}

// Minimal GEGLU test after Q4 matvec in same encoder
// (appended for quick test)
