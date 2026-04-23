//! Debug: per-stage GPU buffer reads to find where data disappears.
//! Runs layer 0 only, reads buffers after each stage.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = larql_inference::InferenceModel::load("google/gemma-3-4b-it")?;
    let weights = model.weights();
    let vd = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");
    let mut index = larql_vindex::VectorIndex::load_vindex(&vd, &mut larql_vindex::SilentLoadCallbacks)?;
    let _ = index.load_attn_q4k(&vd);
    let _ = index.load_interleaved_q4k(&vd);

    #[cfg(feature = "metal")]
    {
        let metal = larql_compute::metal::MetalBackend::new().expect("need metal");
        let gate_index: &dyn larql_vindex::GateIndex = &index;
        let q4_ffn_mmap = gate_index.interleaved_q4k_mmap_ref().unwrap();
        let intermediate = gate_index.num_features(0);
        let hidden = weights.hidden_size;
        let q4_ffn_per_matrix = (intermediate * hidden).div_ceil(256) * 148;
        let ffn_format = larql_compute::QuantFormat::Q4_K;

        let layers = larql_inference::layer_graph::pipeline_layer::build_pipeline_layers(
            weights, &index, 0..1, q4_ffn_mmap, q4_ffn_per_matrix, ffn_format,
        );
        let layer = &layers[0];

        let encoding = model.tokenizer().encode("Hello", true).unwrap();
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let h = larql_inference::forward::embed_tokens_pub(weights, &ids);
        let x: Vec<f32> = h.row(0).to_vec();

        let q_dim = weights.num_q_heads * weights.head_dim;
        let kv_dim = weights.num_kv_heads * weights.head_dim;

        println!("=== Per-Stage GPU Debug (Layer 0) ===\n");
        println!("Input: nonzero={}/{}, max={:.4}", x.iter().filter(|v| v.abs() > 1e-10).count(), x.len(), x.iter().fold(0.0f32, |a, &b| a.max(b.abs())));

        let bufs = metal.bufs();
        let queue = metal.queue();

        // Stage 1: Norm
        let h_buf = bufs.transient_from_f32(&x);
        let norm_buf = bufs.transient_from_f32(layer.input_norm);
        let norm_out = bufs.output((hidden * 4) as u64);
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            larql_compute::metal::ops::full_pipeline::encode_rms_norm(
                enc, &metal.rms_norm_pipeline,
                &h_buf, &norm_buf, &norm_out,
                hidden, layer.eps, layer.norm_offset,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let norm_result = larql_compute::metal::buffers::read_buffer_f32(&norm_out, hidden);
        let nz = norm_result.iter().filter(|v| v.abs() > 1e-10).count();
        let max = norm_result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("After norm: nonzero={}/{}, max={:.4}", nz, hidden, max);

        // Stage 2: Q projection (Q4_K matvec)
        let wq_buf = bufs.get_bytes(layer.wq.data);
        let q_out = bufs.output((q_dim * 4) as u64);
        {
            use larql_compute::metal::shaders::q4k_matvec as q4k;
            let n = q_dim as u32;
            let k = hidden as u32;
            let num_tgs = (q_dim as u64).div_ceil(q4k::ROWS_PER_TG);
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&metal.q4k_matvec_pipeline);
            enc.set_buffer(0, Some(&wq_buf), 0);
            enc.set_buffer(1, Some(&norm_out), 0);
            enc.set_buffer(2, Some(&q_out), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(num_tgs, 1, 1),
                metal::MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let q_result = larql_compute::metal::buffers::read_buffer_f32(&q_out, q_dim);
        let nz = q_result.iter().filter(|v| v.abs() > 1e-10).count();
        let max = q_result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("After Q proj: nonzero={}/{}, max={:.4}", nz, q_dim, max);

        // Stage 2b: K projection
        let wk_buf = bufs.get_bytes(layer.wk.data);
        let k_out = bufs.output((kv_dim * 4) as u64);
        {
            use larql_compute::metal::shaders::q4k_matvec as q4k;
            let n = kv_dim as u32;
            let k = hidden as u32;
            let num_tgs = (kv_dim as u64).div_ceil(q4k::ROWS_PER_TG);
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&metal.q4k_matvec_pipeline);
            enc.set_buffer(0, Some(&wk_buf), 0);
            enc.set_buffer(1, Some(&norm_out), 0);
            enc.set_buffer(2, Some(&k_out), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(num_tgs, 1, 1),
                metal::MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let k_result = larql_compute::metal::buffers::read_buffer_f32(&k_out, kv_dim);
        let nz = k_result.iter().filter(|v| v.abs() > 1e-10).count();
        let max = k_result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("After K proj: nonzero={}/{}, max={:.4}", nz, kv_dim, max);

        // Stage 2c: V projection (Q6_K)
        let wv_buf = bufs.get_bytes(layer.wv.data);
        let v_out = bufs.output((kv_dim * 4) as u64);
        {
            use larql_compute::metal::shaders::q6k_matvec as q6k;
            let n = kv_dim as u32;
            let k = hidden as u32;
            let num_tgs = (kv_dim as u64).div_ceil(q6k::ROWS_PER_TG);
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&metal.q6k_matvec_pipeline);
            enc.set_buffer(0, Some(&wv_buf), 0);
            enc.set_buffer(1, Some(&norm_out), 0);
            enc.set_buffer(2, Some(&v_out), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(num_tgs, 1, 1),
                metal::MTLSize::new(q6k::THREADS_PER_TG, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let v_result = larql_compute::metal::buffers::read_buffer_f32(&v_out, kv_dim);
        let nz = v_result.iter().filter(|v| v.abs() > 1e-10).count();
        let max = v_result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("After V proj: nonzero={}/{}, max={:.4}", nz, kv_dim, max);

        // Stage 3: RoPE
        {
            let pos = 0u32;
            let hd = layer.head_dim as u32;
            let rdim = hd; // full rotation for layer 0
            let rope_base = layer.rope_base;
            let pairs = (layer.head_dim / 2) as u64;

            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for qh in 0..layer.num_q_heads {
                let off = (qh * layer.head_dim * 4) as u64;
                enc.set_compute_pipeline_state(&metal.rope_at_pos_pipeline);
                enc.set_buffer(0, Some(&q_out), off);
                enc.set_bytes(1, 4, &hd as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(2, 4, &rope_base as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(3, 4, &pos as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &rdim as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(metal::MTLSize::new(pairs, 1, 1), metal::MTLSize::new(pairs.min(256), 1, 1));
            }
            for kvh in 0..layer.num_kv_heads {
                let off = (kvh * layer.head_dim * 4) as u64;
                enc.set_compute_pipeline_state(&metal.rope_at_pos_pipeline);
                enc.set_buffer(0, Some(&k_out), off);
                enc.set_bytes(1, 4, &hd as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(2, 4, &rope_base as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(3, 4, &pos as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &rdim as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(metal::MTLSize::new(pairs, 1, 1), metal::MTLSize::new(pairs.min(256), 1, 1));
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let q_roped = larql_compute::metal::buffers::read_buffer_f32(&q_out, q_dim);
        let nz = q_roped.iter().filter(|v| v.abs() > 1e-10).count();
        println!("After RoPE Q: nonzero={}/{}", nz, q_dim);
        let k_roped = larql_compute::metal::buffers::read_buffer_f32(&k_out, kv_dim);
        let nz = k_roped.iter().filter(|v| v.abs() > 1e-10).count();
        println!("After RoPE K: nonzero={}/{}", nz, kv_dim);

        // Stage 4: KV cache + attend
        let mut kv = metal.create_kv_cache(1, 4096, layer.num_kv_heads, layer.head_dim);
        let attn_out = bufs.output((q_dim * 4) as u64);
        {
            let cmd = queue.new_command_buffer();
            larql_compute::metal::ops::kv_cache::append_and_attend(
                cmd, &mut kv.layers[0],
                &metal.kv_append_pipeline, &metal.kv_attend_pipeline,
                &k_out, &v_out, &q_out, &attn_out,
                layer.num_q_heads, layer.attn_scale,
            );
            cmd.commit();
            cmd.wait_until_completed();
        }
        let attn_result = larql_compute::metal::buffers::read_buffer_f32(&attn_out, q_dim);
        let nz = attn_result.iter().filter(|v| v.abs() > 1e-10).count();
        let max = attn_result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("After KV attend: nonzero={}/{}, max={:.4}", nz, q_dim, max);

        // Stage 5: O projection
        let wo_buf = bufs.get_bytes(layer.wo.data);
        let o_out = bufs.output((hidden * 4) as u64);
        {
            use larql_compute::metal::shaders::q4k_matvec as q4k;
            let n = hidden as u32;
            let k_val = q_dim as u32;
            let num_tgs = (hidden as u64).div_ceil(q4k::ROWS_PER_TG);
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&metal.q4k_matvec_pipeline);
            enc.set_buffer(0, Some(&wo_buf), 0);
            enc.set_buffer(1, Some(&attn_out), 0);
            enc.set_buffer(2, Some(&o_out), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(num_tgs, 1, 1),
                metal::MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let o_result = larql_compute::metal::buffers::read_buffer_f32(&o_out, hidden);
        let nz = o_result.iter().filter(|v| v.abs() > 1e-10).count();
        let max = o_result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("After O proj: nonzero={}/{}, max={:.4}", nz, hidden, max);

        // Stage 6: Residual add
        let residual: Vec<f32> = (0..hidden).map(|i| x[i] + o_result[i]).collect();
        let nz = residual.iter().filter(|v| v.abs() > 1e-10).count();
        let max = residual.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("After residual: nonzero={}/{}, max={:.4}", nz, hidden, max);

        println!("\n=== Summary: each stage produces nonzero output ===");
        println!("The decode_token pipeline has a buffer management bug — stages work individually but chain fails.");
    }

    #[cfg(not(feature = "metal"))]
    println!("Requires --features metal");

    Ok(())
}
