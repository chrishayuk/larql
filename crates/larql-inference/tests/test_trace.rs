#![cfg(not(target_arch = "wasm32"))]

//! Integration tests for the trace module: TraceStore, BoundaryStore, ContextStore.
//!
//! These are format/IO tests using synthetic data — no real model needed.

use tempfile::TempDir;

/// Generate a deterministic f32 vector of `len` elements seeded by `seed`.
fn synth_vec(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| (seed as f32 * 0.1 + i as f32 * 0.01).sin())
        .collect()
}

mod test_trace_store {
    use super::*;
    use larql_inference::{TraceNode, TraceStore, TraceWriter};

    const HIDDEN: usize = 16;
    const N_LAYERS: usize = 4;

    fn make_chain(position: usize, seed_base: u32) -> Vec<TraceNode> {
        // n_layers + 1 nodes: embedding (layer -1) + transformer layers 0..n_layers-1
        (0..=N_LAYERS as i32)
            .enumerate()
            .map(|(store_idx, layer)| {
                let s = seed_base + store_idx as u32;
                TraceNode {
                    layer: layer - 1, // -1 for embedding, 0..N_LAYERS-1 for layers
                    position,
                    residual: synth_vec(HIDDEN, s * 3),
                    attn_delta: synth_vec(HIDDEN, s * 3 + 1),
                    ffn_delta: synth_vec(HIDDEN, s * 3 + 2),
                }
            })
            .collect()
    }

    #[test]
    fn write_chain_read_back_exact_match() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("trace.bin");

        // Write two token chains
        let chain0 = make_chain(0, 100);
        let chain1 = make_chain(1, 200);

        {
            let mut writer = TraceWriter::create(&path, HIDDEN, N_LAYERS).unwrap();
            writer.append_chain(&chain0).unwrap();
            writer.append_chain(&chain1).unwrap();
            assert_eq!(writer.n_tokens(), 2);
            writer.finish().unwrap();
        }

        // Read back
        let store = TraceStore::open(&path).unwrap();
        assert_eq!(store.n_tokens(), 2);
        assert_eq!(store.n_layers(), N_LAYERS);
        assert_eq!(store.hidden_size(), HIDDEN);

        // Verify every vector in chain 0
        let n_waypoints = N_LAYERS + 1;
        for (layer_idx, expected) in chain0.iter().enumerate().take(n_waypoints) {
            let got_residual = store.residual(0, layer_idx).unwrap();
            let got_attn = store.attn_delta(0, layer_idx).unwrap();
            let got_ffn = store.ffn_delta(0, layer_idx).unwrap();

            assert_eq!(got_residual.len(), HIDDEN);
            assert_eq!(got_residual, expected.residual.as_slice());
            assert_eq!(got_attn, expected.attn_delta.as_slice());
            assert_eq!(got_ffn, expected.ffn_delta.as_slice());
        }

        // Verify chain 1
        for (layer_idx, expected) in chain1.iter().enumerate().take(n_waypoints) {
            let got_residual = store.residual(1, layer_idx).unwrap();
            assert_eq!(got_residual, expected.residual.as_slice());
        }
    }

    #[test]
    fn out_of_bounds_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("trace_oob.bin");

        let chain = make_chain(0, 42);
        {
            let mut writer = TraceWriter::create(&path, HIDDEN, N_LAYERS).unwrap();
            writer.append_chain(&chain).unwrap();
            writer.finish().unwrap();
        }

        let store = TraceStore::open(&path).unwrap();
        // Token index out of bounds
        assert!(store.residual(1, 0).is_none());
        // Layer index out of bounds
        assert!(store.residual(0, N_LAYERS + 1).is_none());
        // Component index out of bounds via read_vector
        assert!(store.read_vector(0, 0, 3).is_none());
    }

    #[test]
    fn node_method_reconstructs_trace_node() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("trace_node.bin");

        let chain = make_chain(0, 77);
        {
            let mut writer = TraceWriter::create(&path, HIDDEN, N_LAYERS).unwrap();
            writer.append_chain(&chain).unwrap();
            writer.finish().unwrap();
        }

        let store = TraceStore::open(&path).unwrap();
        // Layer 0 in the store = embedding = layer -1 in TraceNode
        let node = store.node(0, 0).unwrap();
        assert_eq!(node.layer, -1);
        assert_eq!(node.position, 0);
        assert_eq!(node.residual, chain[0].residual);
        assert_eq!(node.attn_delta, chain[0].attn_delta);
        assert_eq!(node.ffn_delta, chain[0].ffn_delta);
    }

    #[test]
    fn wrong_chain_length_rejected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("trace_bad.bin");

        let mut writer = TraceWriter::create(&path, HIDDEN, N_LAYERS).unwrap();
        // Pass too few nodes
        let short_chain: Vec<TraceNode> = make_chain(0, 1).into_iter().take(2).collect();
        let result = writer.append_chain(&short_chain);
        assert!(result.is_err());
    }
}

mod test_boundary_store {
    use super::*;
    use larql_inference::{BoundaryStore, BoundaryWriter};

    const HIDDEN: usize = 32;
    const WINDOW: usize = 200;

    #[test]
    fn append_and_read_back() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("boundary.bin");

        let r0 = synth_vec(HIDDEN, 10);
        let r1 = synth_vec(HIDDEN, 20);
        let r2 = synth_vec(HIDDEN, 30);

        {
            let mut writer = BoundaryWriter::create(&path, HIDDEN, WINDOW, 100).unwrap();
            writer.append(0, WINDOW, &r0).unwrap();
            writer.append(WINDOW, WINDOW, &r1).unwrap();
            writer.append(WINDOW * 2, WINDOW, &r2).unwrap();
            assert_eq!(writer.n_boundaries(), 3);
            assert_eq!(writer.total_tokens(), WINDOW * 3);
            writer.finish().unwrap();
        }

        let store = BoundaryStore::open(&path).unwrap();
        assert_eq!(store.n_boundaries(), 3);
        assert_eq!(store.total_tokens(), WINDOW * 3);
        assert_eq!(store.hidden_size(), HIDDEN);
        assert_eq!(store.window_size(), WINDOW);

        // Exact match
        assert_eq!(store.residual(0).unwrap(), r0.as_slice());
        assert_eq!(store.residual(1).unwrap(), r1.as_slice());
        assert_eq!(store.residual(2).unwrap(), r2.as_slice());

        // Out of bounds
        assert!(store.residual(3).is_none());
    }

    #[test]
    fn boundary_for_token_lookup() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("boundary_lookup.bin");

        {
            let mut writer = BoundaryWriter::create(&path, HIDDEN, WINDOW, 100).unwrap();
            writer.append(0, 200, &synth_vec(HIDDEN, 1)).unwrap();
            writer.append(200, 200, &synth_vec(HIDDEN, 2)).unwrap();
            writer.finish().unwrap();
        }

        let store = BoundaryStore::open(&path).unwrap();
        assert_eq!(store.boundary_for_token(0), Some(0));
        assert_eq!(store.boundary_for_token(199), Some(0));
        assert_eq!(store.boundary_for_token(200), Some(1));
        assert_eq!(store.boundary_for_token(399), Some(1));
        assert_eq!(store.boundary_for_token(400), None);
    }

    #[test]
    fn token_range() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("boundary_range.bin");

        {
            let mut writer = BoundaryWriter::create(&path, HIDDEN, WINDOW, 100).unwrap();
            writer.append(0, 150, &synth_vec(HIDDEN, 1)).unwrap();
            writer.append(150, 200, &synth_vec(HIDDEN, 2)).unwrap();
            writer.finish().unwrap();
        }

        let store = BoundaryStore::open(&path).unwrap();
        assert_eq!(store.token_range(0), Some((0, 150)));
        assert_eq!(store.token_range(1), Some((150, 350)));
        assert_eq!(store.token_range(2), None);
    }

    #[test]
    fn size_mismatch_rejected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("boundary_bad.bin");

        let mut writer = BoundaryWriter::create(&path, HIDDEN, WINDOW, 100).unwrap();
        let bad = synth_vec(HIDDEN + 1, 1); // wrong size
        let result = writer.append(0, 200, &bad);
        assert!(result.is_err());
    }
}

mod test_context_store {
    use super::*;
    use larql_inference::{ContextStore, ContextTier, ContextWriter};

    const HIDDEN: usize = 16;
    const N_LAYERS: usize = 8;
    const WINDOW: usize = 100;
    const CRITICAL: &[usize] = &[2, 5, 7];

    #[test]
    fn tier1_residual_only() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("ctx_t1.bin");

        let r0 = synth_vec(HIDDEN, 10);
        let r1 = synth_vec(HIDDEN, 20);

        {
            let mut writer = ContextWriter::create(
                &path,
                HIDDEN,
                N_LAYERS,
                WINDOW,
                ContextTier::Residual,
                CRITICAL,
                100,
            )
            .unwrap();
            writer.append(0, WINDOW, &r0, &[], &[]).unwrap();
            writer.append(WINDOW, WINDOW, &r1, &[], &[]).unwrap();
            assert_eq!(writer.n_boundaries(), 2);
            writer.finish().unwrap();
        }

        let store = ContextStore::open(&path).unwrap();
        assert_eq!(store.n_boundaries(), 2);
        assert_eq!(store.tier(), ContextTier::Residual);
        assert_eq!(store.hidden_size(), HIDDEN);
        assert_eq!(store.window_size(), WINDOW);
        assert_eq!(store.critical_layers(), vec![2, 5, 7]);

        assert_eq!(store.residual(0).unwrap(), r0.as_slice());
        assert_eq!(store.residual(1).unwrap(), r1.as_slice());

        // FFN/attn deltas not available at Tier 1
        assert!(store.ffn_delta(0, 0).is_none());
        assert!(store.attn_delta(0, 0).is_none());
    }

    #[test]
    fn tier2_residual_plus_ffn_deltas() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("ctx_t2.bin");

        let residual = synth_vec(HIDDEN, 50);
        let ffn_deltas: Vec<Vec<f32>> = (0..CRITICAL.len())
            .map(|i| synth_vec(HIDDEN, 100 + i as u32))
            .collect();

        {
            let mut writer = ContextWriter::create(
                &path,
                HIDDEN,
                N_LAYERS,
                WINDOW,
                ContextTier::FfnDeltas,
                CRITICAL,
                100,
            )
            .unwrap();
            writer
                .append(0, WINDOW, &residual, &ffn_deltas, &[])
                .unwrap();
            writer.finish().unwrap();
        }

        let store = ContextStore::open(&path).unwrap();
        assert_eq!(store.tier(), ContextTier::FfnDeltas);
        assert_eq!(store.residual(0).unwrap(), residual.as_slice());

        for (i, expected) in ffn_deltas.iter().enumerate() {
            assert_eq!(store.ffn_delta(0, i).unwrap(), expected.as_slice());
        }

        // Attn deltas not available at Tier 2
        assert!(store.attn_delta(0, 0).is_none());
    }

    #[test]
    fn tier3_full_store() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("ctx_t3.bin");

        let residual = synth_vec(HIDDEN, 70);
        let ffn_deltas: Vec<Vec<f32>> = (0..CRITICAL.len())
            .map(|i| synth_vec(HIDDEN, 200 + i as u32))
            .collect();
        let attn_deltas: Vec<Vec<f32>> = (0..CRITICAL.len())
            .map(|i| synth_vec(HIDDEN, 300 + i as u32))
            .collect();

        {
            let mut writer = ContextWriter::create(
                &path,
                HIDDEN,
                N_LAYERS,
                WINDOW,
                ContextTier::Full,
                CRITICAL,
                100,
            )
            .unwrap();
            writer
                .append(0, WINDOW, &residual, &ffn_deltas, &attn_deltas)
                .unwrap();
            writer.finish().unwrap();
        }

        let store = ContextStore::open(&path).unwrap();
        assert_eq!(store.tier(), ContextTier::Full);
        assert_eq!(store.residual(0).unwrap(), residual.as_slice());

        for (i, expected) in ffn_deltas.iter().enumerate() {
            assert_eq!(store.ffn_delta(0, i).unwrap(), expected.as_slice());
        }
        for (i, expected) in attn_deltas.iter().enumerate() {
            assert_eq!(store.attn_delta(0, i).unwrap(), expected.as_slice());
        }
    }

    #[test]
    fn context_boundary_for_token() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("ctx_lookup.bin");

        {
            let mut writer = ContextWriter::create(
                &path,
                HIDDEN,
                N_LAYERS,
                WINDOW,
                ContextTier::Residual,
                CRITICAL,
                100,
            )
            .unwrap();
            writer
                .append(0, 100, &synth_vec(HIDDEN, 1), &[], &[])
                .unwrap();
            writer
                .append(100, 100, &synth_vec(HIDDEN, 2), &[], &[])
                .unwrap();
            writer.finish().unwrap();
        }

        let store = ContextStore::open(&path).unwrap();
        assert_eq!(store.boundary_for_token(50), Some(0));
        assert_eq!(store.boundary_for_token(150), Some(1));
        assert_eq!(store.boundary_for_token(200), None);
    }

    #[test]
    fn bytes_per_boundary_matches_tier() {
        let dir = TempDir::new().unwrap();

        // Tier 1: 1 vector
        let path1 = dir.path().join("ctx_bpb1.bin");
        {
            let mut w = ContextWriter::create(
                &path1,
                HIDDEN,
                N_LAYERS,
                WINDOW,
                ContextTier::Residual,
                CRITICAL,
                10,
            )
            .unwrap();
            w.append(0, WINDOW, &synth_vec(HIDDEN, 1), &[], &[])
                .unwrap();
            w.finish().unwrap();
        }
        let s1 = ContextStore::open(&path1).unwrap();
        assert_eq!(s1.bytes_per_boundary(), HIDDEN * 4); // 1 vector

        // Tier 2: 1 + n_critical vectors
        let path2 = dir.path().join("ctx_bpb2.bin");
        {
            let ffn: Vec<Vec<f32>> = (0..3).map(|i| synth_vec(HIDDEN, 10 + i)).collect();
            let mut w = ContextWriter::create(
                &path2,
                HIDDEN,
                N_LAYERS,
                WINDOW,
                ContextTier::FfnDeltas,
                CRITICAL,
                10,
            )
            .unwrap();
            w.append(0, WINDOW, &synth_vec(HIDDEN, 1), &ffn, &[])
                .unwrap();
            w.finish().unwrap();
        }
        let s2 = ContextStore::open(&path2).unwrap();
        assert_eq!(s2.bytes_per_boundary(), (1 + CRITICAL.len()) * HIDDEN * 4);

        // Tier 3: 1 + 2*n_critical vectors
        let path3 = dir.path().join("ctx_bpb3.bin");
        {
            let ffn: Vec<Vec<f32>> = (0..3).map(|i| synth_vec(HIDDEN, 20 + i)).collect();
            let attn: Vec<Vec<f32>> = (0..3).map(|i| synth_vec(HIDDEN, 30 + i)).collect();
            let mut w = ContextWriter::create(
                &path3,
                HIDDEN,
                N_LAYERS,
                WINDOW,
                ContextTier::Full,
                CRITICAL,
                10,
            )
            .unwrap();
            w.append(0, WINDOW, &synth_vec(HIDDEN, 1), &ffn, &attn)
                .unwrap();
            w.finish().unwrap();
        }
        let s3 = ContextStore::open(&path3).unwrap();
        assert_eq!(
            s3.bytes_per_boundary(),
            (1 + 2 * CRITICAL.len()) * HIDDEN * 4
        );
    }
}

mod test_additive_property {
    use super::*;
    use larql_inference::{TraceNode, TraceStore, TraceWriter};

    const HIDDEN: usize = 8;
    const N_LAYERS: usize = 3;

    /// Build a chain where residual[layer] = residual[layer-1] + attn_delta[layer] + ffn_delta[layer].
    /// This mirrors the actual residual stream computation in a transformer.
    fn make_additive_chain() -> Vec<TraceNode> {
        let mut nodes = Vec::new();

        // Embedding layer (layer -1, store index 0): initial residual, zero deltas
        let emb_residual = synth_vec(HIDDEN, 999);
        nodes.push(TraceNode {
            layer: -1,
            position: 0,
            residual: emb_residual.clone(),
            attn_delta: vec![0.0; HIDDEN],
            ffn_delta: vec![0.0; HIDDEN],
        });

        let mut prev_residual = emb_residual;

        for layer in 0..N_LAYERS as i32 {
            let attn = synth_vec(HIDDEN, (layer as u32 + 1) * 10);
            let ffn = synth_vec(HIDDEN, (layer as u32 + 1) * 10 + 5);
            let residual: Vec<f32> = (0..HIDDEN)
                .map(|i| prev_residual[i] + attn[i] + ffn[i])
                .collect();

            nodes.push(TraceNode {
                layer,
                position: 0,
                residual: residual.clone(),
                attn_delta: attn,
                ffn_delta: ffn,
            });
            prev_residual = residual;
        }

        nodes
    }

    #[test]
    fn residual_equals_prev_plus_attn_plus_ffn() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("trace_additive.bin");

        let chain = make_additive_chain();

        {
            let mut writer = TraceWriter::create(&path, HIDDEN, N_LAYERS).unwrap();
            writer.append_chain(&chain).unwrap();
            writer.finish().unwrap();
        }

        let store = TraceStore::open(&path).unwrap();

        // For each transformer layer, verify the additive property
        for layer_idx in 1..=N_LAYERS {
            let prev_residual = store.residual(0, layer_idx - 1).unwrap();
            let attn_delta = store.attn_delta(0, layer_idx).unwrap();
            let ffn_delta = store.ffn_delta(0, layer_idx).unwrap();
            let residual = store.residual(0, layer_idx).unwrap();

            for i in 0..HIDDEN {
                let expected = prev_residual[i] + attn_delta[i] + ffn_delta[i];
                assert!(
                    (residual[i] - expected).abs() < 1e-6,
                    "layer {} dim {}: {} != {} + {} + {}",
                    layer_idx,
                    i,
                    residual[i],
                    prev_residual[i],
                    attn_delta[i],
                    ffn_delta[i],
                );
            }
        }
    }
}
