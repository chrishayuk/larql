//! Records → metrics. Forecasts are COMPUTED and written, never asserted:
//! a falsified forecast is a result. Only instrument validity (foreign
//! allocations, unresolved capacities, unclassified traffic) is reported
//! as `instrument_ok`, and the controls assert on it.

use serde_json::{json, Value};

use larql_vindex::format::vindex3::opplan::exec::continuation::LayerContinuationGeometry;

use super::measured::{Backing, CallRecord, IntervalRecord, Method, Window};

#[derive(Clone, Copy, Debug, Default)]
pub struct Storage {
    pub kv_payload: usize,
    pub row_storage: usize,
    pub matrix: usize,
    pub headers: usize,
    pub recurrent_payload: usize,
    pub recurrent_allocated: usize,
    pub latent_payload: usize,
    pub latent_allocated: usize,
    /// Backings whose capacity the live table could not resolve.
    pub unresolved: usize,
}

impl Storage {
    pub fn of(inventory: &[Backing]) -> Self {
        let mut s = Storage::default();
        for b in inventory {
            let bytes = match b.bytes {
                Some(bytes) => bytes,
                None => {
                    // Headers of a state built outside any scope (a
                    // resumed row store's outer Vec) are reported
                    // unresolved; every other kind must resolve.
                    if !b.kind.ends_with("_header") {
                        s.unresolved += 1;
                    }
                    0
                }
            };
            match b.kind {
                "k_row" | "v_row" => {
                    s.kv_payload += b.payload_bytes;
                    s.row_storage += bytes;
                }
                "k_matrix" | "v_matrix" => s.matrix += bytes,
                "k_header" | "v_header" => s.headers += bytes,
                "recurrent" => {
                    s.recurrent_payload += b.payload_bytes;
                    s.recurrent_allocated += bytes;
                }
                "latent_row" => {
                    s.latent_payload += b.payload_bytes;
                    s.latent_allocated += bytes;
                }
                other => panic!("unknown backing kind {other}"),
            }
        }
        s
    }

    /// K/V representation bytes: row allocations + matrix capacity,
    /// headers excluded (M1's quantity).
    pub fn kv_allocated(&self) -> usize {
        self.row_storage + self.matrix
    }

    pub fn json(&self) -> Value {
        json!({
            "kv_payload_bytes": self.kv_payload,
            "kv_row_allocated_bytes": self.row_storage,
            "kv_matrix_allocated_bytes": self.matrix,
            "kv_allocated_bytes_headers_excluded": self.kv_allocated(),
            "header_bytes": self.headers,
            "recurrent_payload_bytes": self.recurrent_payload,
            "recurrent_allocated_bytes": self.recurrent_allocated,
            "latent_payload_bytes": self.latent_payload,
            "latent_allocated_bytes": self.latent_allocated,
            "unresolved_capacities": self.unresolved,
        })
    }
}

/// Append traffic aggregated over a phase.
pub fn appends(calls: &[CallRecord], phase: &str) -> Value {
    let mut n = 0u64;
    let (mut row_sized, mut dup, mut dup_bytes, mut first, mut moved, mut moved_bytes) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut in_place, mut header, mut freed, mut unclassified, mut adopted, mut grew) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut moved_in = 0u64;
    for c in calls
        .iter()
        .filter(|c| c.phase == phase && c.method == Method::Append)
    {
        let t = c.append.expect("append traffic");
        n += 1;
        row_sized += t.row_sized_allocs;
        dup += t.duplicate_allocs;
        dup_bytes += t.duplicate_bytes;
        first += t.matrix_first_alloc_bytes;
        moved += t.matrix_moved;
        moved_bytes += t.matrix_moved_bytes;
        in_place += t.matrix_in_place;
        header += t.header_alloc_bytes;
        freed += t.incoming_freed;
        unclassified += t.unclassified;
        adopted += u64::from(t.adopted);
        grew += u64::from(t.matrix_grew_by_one == Some(true));
        moved_in += c.moved_in_bytes;
    }
    json!({
        "appends": n,
        "row_sized_allocs": row_sized,
        "duplicate_allocs": dup,
        "duplicate_bytes": dup_bytes,
        "matrix_first_alloc_bytes": first,
        "matrix_realloc_moved": moved,
        "matrix_realloc_moved_bytes": moved_bytes,
        "matrix_realloc_in_place": in_place,
        "header_alloc_bytes": header,
        "incoming_rows_freed": freed,
        "unclassified_events": unclassified,
        "adopted_appends": adopted,
        "matrix_grew_by_one": grew,
        "moved_in_capacity_bytes": moved_in,
    })
}

pub fn events_dropped(calls: &[CallRecord], intervals: &[IntervalRecord]) -> bool {
    calls.iter().any(|c| c.delta.events_dropped) || intervals.iter().any(|i| i.delta.events_dropped)
}

/// M3: rows offered by `keys` per layer and phase, beside the structural
/// selected range (start..=position) the kernel's window implies.
pub fn rows_offered(
    calls: &[CallRecord],
    geometry: &[LayerContinuationGeometry],
    phase: &str,
) -> Value {
    let mut layers = Vec::new();
    for (layer, g) in geometry.iter().enumerate() {
        let Some(kv) = g.kv_side() else { continue };
        let keys: Vec<&CallRecord> = calls
            .iter()
            .filter(|c| c.phase == phase && c.method == Method::Keys && c.layer == Some(layer))
            .collect();
        let offered: Vec<usize> = keys.iter().filter_map(|c| c.rows_offered).collect();
        let positions: Vec<usize> = keys.iter().map(|c| c.position).collect();
        if offered.is_empty() {
            continue;
        }
        // `keys` is called before the position's own row is appended, so
        // an offer of h held rows is a step over h + 1 positions.
        let selected: Vec<usize> = offered
            .iter()
            .map(|&h| kv.window.map_or(h + 1, |w| (h + 1).min(w)))
            .collect();
        let n_end = *offered.last().unwrap() + 1;
        let outside = kv
            .window
            .map_or(0.0, |w| (1.0 - w as f64 / n_end as f64).max(0.0));
        layers.push(json!({
            "layer": layer,
            "window": kv.window,
            "keys_calls": offered.len(),
            "positions": positions,
            "rows_offered": offered,
            "kernel_selected_range_len": selected,
            "rows_held_at_end": n_end,
            "held_outside_range_fraction_at_end": outside,
        }));
    }
    Value::Array(layers)
}

/// M4: per conv-QKV call, target (row-sized) allocations in the frozen
/// window (after `values`) and in the D1 union (after `keys` + after
/// `values`), against h.
pub fn conv_qkv_windows(intervals: &[IntervalRecord], phase: &str) -> Value {
    let mut calls = Vec::new();
    let windows: Vec<&IntervalRecord> = intervals
        .iter()
        .filter(|i| i.phase == phase && matches!(i.window, Window::AfterKeys | Window::AfterValues))
        .collect();
    for pair in windows.chunks(2) {
        let [k, v] = pair else {
            calls.push(json!({"unpaired_window": format!("{:?}", pair[0].window)}));
            continue;
        };
        assert_eq!(
            k.window,
            Window::AfterKeys,
            "windows alternate keys then values"
        );
        assert_eq!(
            v.window,
            Window::AfterValues,
            "windows alternate keys then values"
        );
        calls.push(json!({
            "layer": k.layer,
            "h": k.rows_held,
            "row_sized_allocs_frozen_window": v.target_allocs,
            "row_sized_allocs_union": k.target_allocs + v.target_allocs,
            "all_allocs_union": k.delta.allocs + v.delta.allocs,
            "alloc_bytes_union": k.delta.alloc_bytes + v.delta.alloc_bytes,
        }));
    }
    Value::Array(calls)
}

/// M8: per recurrent call, allocations sized as a declared conv-history
/// buffer inside the operator's window.
pub fn recurrent_windows(intervals: &[IntervalRecord], phase: &str) -> Value {
    Value::Array(
        intervals
            .iter()
            .filter(|i| i.phase == phase && i.window == Window::AfterRecurrent)
            .map(|i| {
                json!({
                    "layer": i.layer,
                    "conv_history_buffer_bytes": i.target_bytes,
                    "conv_history_sized_allocs": i.target_allocs,
                    "window_allocs": i.delta.allocs,
                    "foreign": i.delta.foreign,
                })
            })
            .collect(),
    )
}

pub fn geometry_json(geometry: &[LayerContinuationGeometry]) -> Value {
    Value::Array(
        geometry
            .iter()
            .map(|g| {
                json!({
                    "kv_dim": g.kv_side().map(|k| k.kv_dim),
                    "window": g.kv_side().and_then(|k| k.window),
                    "latent_width": g.latent_kv().map(|l| l.width),
                    "recurrent_buffer_bytes": g.recurrent().map(|r| r.buffers.iter().map(|b| b.bytes()).collect::<Vec<_>>()),
                    "recurrent_payload_bytes": g.recurrent().map(|r| r.bytes()),
                })
            })
            .collect(),
    )
}
