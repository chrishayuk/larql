//! Seeding and rolling back the KV cache through the `DecodeBackend`
//! trait.
//!
//! A layer's cache has two counters ([`crate::ops::kv_cache::LayerKVCache`]):
//! `current_len`, how many rows it holds, and `abs_position`, the stream
//! position the next row is rotated at. Every decode step moves both. Two
//! trait entry points moved only the first:
//!
//! - `populate_kv_layer` copies in a prefill computed elsewhere (the CPU
//!   prefill ahead of a Metal decode). It set `current_len = seq_len` and
//!   left `abs_position` alone, so on a fresh cache the first decoded
//!   token was rotated at position 0 behind a prompt of `seq_len` tokens.
//! - `truncate_kv_cache` rolls back speculative rows. It lowered
//!   `current_len` and left `abs_position` ahead by every row dropped,
//!   although the trait documents the next pass as "starting from
//!   position `len`".
//!
//! `populate_kv_layer` also copied `seq_len * heads * head_dim` floats
//! into the layer's buffer without comparing that to the buffer's size.
//!
//! The bulk prefill inside the Metal pipeline (`ops/full_pipeline/kv_copy`)
//! always set the position; it is these two side entrances that did not.
//! The tests below read the counters, because they are what the decode
//! kernels take RoPE's position from (`decode/encode_attn/context.rs`).

#[cfg(test)]
mod tests {
    use crate::MetalBackend;
    use larql_compute::backend::DecodeBackend;

    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    /// Rows the test cache can hold.
    const MAX_SEQ: usize = 8;
    /// Rows of a seeded prompt.
    const PROMPT_ROWS: usize = 5;
    const LAYER: usize = 0;

    fn backend() -> MetalBackend {
        let backend = MetalBackend::new().expect(
            "Metal backend must build: the shader library failed to compile or no device exists",
        );
        backend.preallocate_kv_cache_per_layer(&[(KV_HEADS, HEAD_DIM)], MAX_SEQ);
        backend
    }

    fn rows(count: usize) -> Vec<f32> {
        (0..count * KV_HEADS * HEAD_DIM).map(|i| i as f32).collect()
    }

    fn seed(backend: &MetalBackend, count: usize) {
        let data = rows(count);
        backend.populate_kv_layer(LAYER, &data, &data, count, KV_HEADS, HEAD_DIM);
    }

    /// `(current_len, abs_position)` of the test layer.
    fn counters(backend: &MetalBackend) -> (usize, usize) {
        let guard = backend.kv_cache.lock().expect("kv cache lock");
        let layer = &guard.as_ref().expect("cache installed").layers[LAYER];
        (layer.current_len, layer.abs_position)
    }

    /// The panic message of `f`, or `None` if it returned.
    fn refusal(f: impl FnOnce() + std::panic::UnwindSafe) -> Option<String> {
        let payload = std::panic::catch_unwind(f).err()?;
        payload.downcast_ref::<String>().cloned()
    }

    #[test]
    fn a_seeded_cache_is_positioned_after_its_rows() {
        let backend = backend();

        seed(&backend, PROMPT_ROWS);

        assert_eq!(counters(&backend), (PROMPT_ROWS, PROMPT_ROWS));
    }

    #[test]
    fn a_reseed_replaces_the_position_as_well_as_the_rows() {
        let backend = backend();
        seed(&backend, PROMPT_ROWS);

        seed(&backend, 2);

        assert_eq!(counters(&backend), (2, 2));
    }

    #[test]
    fn a_rollback_moves_the_position_back_with_the_rows() {
        let backend = backend();
        seed(&backend, PROMPT_ROWS);

        backend.truncate_kv_cache(3);

        assert_eq!(counters(&backend), (3, 3));
    }

    #[test]
    fn a_rollback_past_the_end_changes_nothing() {
        // The rows beyond the occupancy were never written.
        let backend = backend();
        seed(&backend, PROMPT_ROWS);

        backend.truncate_kv_cache(PROMPT_ROWS + 2);

        assert_eq!(counters(&backend), (PROMPT_ROWS, PROMPT_ROWS));
    }

    #[test]
    fn a_rollback_keeps_the_offset_a_slid_window_left() {
        // After a window slides, occupancy trails the position. A rollback
        // of one row takes one from each.
        let backend = backend();
        seed(&backend, PROMPT_ROWS);
        backend.compact_kv_to_window(1);
        let (len, position) = counters(&backend);
        assert!(
            len < position,
            "the window slid: {len} rows at position {position}"
        );

        {
            let mut guard = backend.kv_cache.lock().expect("kv cache lock");
            guard.as_mut().expect("cache installed").layers[LAYER].advance_one();
        }
        backend.truncate_kv_cache(len);

        assert_eq!(counters(&backend), (len, position));
    }

    #[test]
    fn a_seed_longer_than_the_cache_is_refused_and_leaves_it_usable() {
        let backend = backend();
        seed(&backend, PROMPT_ROWS);

        let message = refusal(std::panic::AssertUnwindSafe(|| seed(&backend, MAX_SEQ + 1)))
            .expect("a seed that does not fit must be refused");

        assert!(message.contains("at most"), "{message}");
        assert_eq!(
            counters(&backend),
            (PROMPT_ROWS, PROMPT_ROWS),
            "the refusal changed nothing and did not poison the cache"
        );
    }

    #[test]
    fn a_seed_of_another_head_shape_is_refused() {
        let backend = backend();
        let wide = vec![0.0f32; 2 * KV_HEADS * (HEAD_DIM * 2)];

        let message = refusal(std::panic::AssertUnwindSafe(|| {
            backend.populate_kv_layer(LAYER, &wide, &wide, 2, KV_HEADS, HEAD_DIM * 2);
        }))
        .expect("a seed of another shape must be refused");

        assert!(message.contains("shape"), "{message}");
    }

    #[test]
    fn a_seed_whose_data_is_not_its_stated_rows_is_refused() {
        let backend = backend();
        let short = rows(1);

        let message = refusal(std::panic::AssertUnwindSafe(|| {
            backend.populate_kv_layer(LAYER, &short, &short, 2, KV_HEADS, HEAD_DIM);
        }))
        .expect("a seed shorter than its stated rows must be refused");

        assert!(message.contains("floats"), "{message}");
    }
}
