//! CONTINUATION-MEM-1 I1: `Measured<P>`, a provider that IS `P` and records
//! every call — no trait change.
//!
//! Each delegated call runs inside one allocator scope; the record is
//! pushed only after the scope has closed, so the recorder's own
//! allocations are never attributed. On a conv-QKV layer the wrapper also
//! opens the D1 interval windows (`keys` return → next entry, `values`
//! return → next entry), which contain only executor code.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use larql_kv::CanonicalKvState;
use larql_vindex::format::vindex3::opplan::exec::continuation::{
    LatentKvRows, LayerContinuationGeometry, RecurrentState,
};
use larql_vindex::format::vindex3::opplan::exec::kv::{
    ContinuationError, ContinuationProvider, LayerKvGeometry, RowKvState,
};

use super::alloc::{self, Event, EventKind, Scope, ScopeDelta};

const F32_BYTES: usize = std::mem::size_of::<f32>();

/// What the harness can see of a provider's storage beyond the trait.
pub trait Inspect: ContinuationProvider {
    /// The K and V matrix data pointers of `layer`, for a provider whose
    /// authority is a matrix.
    fn matrix_ptrs(&self, layer: usize) -> Option<(usize, usize)>;
    /// Rows held in `layer`'s matrix.
    fn matrix_rows(&self, layer: usize) -> Option<usize>;
}

impl Inspect for RowKvState {
    fn matrix_ptrs(&self, _: usize) -> Option<(usize, usize)> {
        None
    }
    fn matrix_rows(&self, _: usize) -> Option<usize> {
        None
    }
}

impl Inspect for CanonicalKvState {
    fn matrix_ptrs(&self, layer: usize) -> Option<(usize, usize)> {
        let (k, v) = self.cache().get_layer(layer)?;
        Some((k.as_ptr() as usize, v.as_ptr() as usize))
    }
    fn matrix_rows(&self, layer: usize) -> Option<usize> {
        self.cache().get_layer(layer).map(|(k, _)| k.shape()[0])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Prepare,
    PrepareContinuation,
    Append,
    Keys,
    Values,
    Position,
    SetPosition,
    Recurrent,
    Latent,
}

/// Where each event of an append scope went (D4: pointer identity).
#[derive(Clone, Copy, Debug, Default)]
pub struct AppendTraffic {
    pub row_sized_allocs: u64,
    pub duplicate_allocs: u64,
    pub duplicate_bytes: u64,
    pub matrix_first_alloc_bytes: u64,
    pub matrix_moved: u64,
    pub matrix_moved_bytes: u64,
    pub matrix_in_place: u64,
    pub header_alloc_bytes: u64,
    pub incoming_freed: u64,
    pub unclassified: u64,
    /// The stored K row's address equals the moved-in K row's.
    pub adopted: bool,
    /// Matrix row count grew by exactly one.
    pub matrix_grew_by_one: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct CallRecord {
    pub phase: &'static str,
    pub method: Method,
    pub layer: Option<usize>,
    pub rows_offered: Option<usize>,
    pub position: usize,
    pub delta: ScopeDelta,
    pub append: Option<AppendTraffic>,
    pub moved_in_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    /// D1: `keys` returned on a conv-QKV layer.
    AfterKeys,
    /// The frozen window: `values` returned on a conv-QKV layer.
    AfterValues,
    /// M8: `recurrent_state` returned (operator runs in the window).
    AfterRecurrent,
}

#[derive(Clone, Debug)]
pub struct IntervalRecord {
    pub phase: &'static str,
    pub window: Window,
    pub layer: usize,
    /// h: rows the provider held when the window opened.
    pub rows_held: usize,
    pub delta: ScopeDelta,
    /// Allocations whose size is one of `target_bytes`.
    pub target_bytes: Vec<usize>,
    pub target_allocs: u64,
}

/// One backing allocation of continuation storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Backing {
    pub kind: &'static str,
    pub layer: usize,
    pub index: usize,
    pub ptr: usize,
    /// Capacity in bytes; `None` when the live table does not hold it.
    pub bytes: Option<usize>,
    pub payload_bytes: usize,
}

pub type InventorySink = Arc<Mutex<Option<Vec<Backing>>>>;

struct Overlay {
    layer: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

pub struct Measured<P> {
    pub inner: P,
    phase: Cell<&'static str>,
    calls: RefCell<Vec<CallRecord>>,
    intervals: RefCell<Vec<IntervalRecord>>,
    open: RefCell<Option<(Scope, Window, usize, usize, Vec<usize>)>>,
    geometry: Vec<LayerContinuationGeometry>,
    conv_qkv: Vec<bool>,
    recurrent_target: Vec<Option<Vec<usize>>>,
    watch_recurrent: bool,
    overlay: Option<Overlay>,
    sink: Option<(Arc<AtomicBool>, InventorySink)>,
}

impl<P: Inspect> Measured<P> {
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            phase: Cell::new("unset"),
            calls: RefCell::new(Vec::with_capacity(1 << 16)),
            intervals: RefCell::new(Vec::with_capacity(1 << 12)),
            open: RefCell::new(None),
            geometry: Vec::new(),
            conv_qkv: Vec::new(),
            recurrent_target: Vec::new(),
            watch_recurrent: false,
            overlay: None,
            sink: None,
        }
    }

    /// Mark the conv-QKV layers (their KV rows and a recurrent history
    /// share one provider): D1 windows open on these.
    pub fn with_conv_qkv(mut self, layers: Vec<bool>) -> Self {
        self.conv_qkv = layers;
        self
    }

    /// M8: open a window after each `recurrent_state` and count
    /// allocations of the layer's declared conv-history payload.
    pub fn watching_recurrent(mut self, conv_history_bytes: Vec<Option<Vec<usize>>>) -> Self {
        self.recurrent_target = conv_history_bytes;
        self.watch_recurrent = true;
        self
    }

    // Used by the larql-server handoff target, which includes this file.
    #[allow(dead_code)]
    pub fn with_sink(mut self, trigger: Arc<AtomicBool>, sink: InventorySink) -> Self {
        self.sink = Some((trigger, sink));
        self
    }

    pub fn set_phase(&self, phase: &'static str) {
        self.close_window();
        self.phase.set(phase);
    }

    /// Close any open window and hand back the records.
    pub fn take_records(&self) -> (Vec<CallRecord>, Vec<IntervalRecord>) {
        self.close_window();
        (
            std::mem::take(&mut self.calls.borrow_mut()),
            std::mem::take(&mut self.intervals.borrow_mut()),
        )
    }

    /// I2: overlay `layer`'s rows with a copy whose `row` is perturbed in
    /// its largest-magnitude cell by a factor (1 + 2⁻⁴), K and V both (D2).
    pub fn perturb(&mut self, layer: usize, row: usize) {
        let mut keys = self.inner.keys(layer).to_vec();
        let mut values = self.inner.values(layer).to_vec();
        scale_largest(&mut keys[row]);
        scale_largest(&mut values[row]);
        self.overlay = Some(Overlay {
            layer,
            keys,
            values,
        });
    }

    /// Every backing allocation of continuation storage, with capacity.
    pub fn inventory(&mut self) -> Vec<Backing> {
        inventory_of(&mut self.inner, &self.geometry)
    }

    fn record(&self, method: Method, layer: Option<usize>, rows: Option<usize>, delta: ScopeDelta) {
        let position = self.inner.position();
        self.calls.borrow_mut().push(CallRecord {
            phase: self.phase.get(),
            method,
            layer,
            rows_offered: rows,
            position,
            delta,
            append: None,
            moved_in_bytes: 0,
        });
    }

    fn row_bytes(&self, layer: usize) -> usize {
        self.geometry
            .get(layer)
            .and_then(|g| g.kv_side())
            .map_or(0, |g| g.kv_dim * F32_BYTES)
    }
}

fn scale_largest(row: &mut [f32]) {
    let (i, _) = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .expect("a row has cells");
    assert!(row[i] != 0.0, "D2: never perturb a zero cell");
    row[i] *= 1.0 + 1.0 / 16.0;
}

fn count_allocs_of(events: &[Event], bytes: &[usize]) -> u64 {
    events
        .iter()
        .filter(|e| e.kind == EventKind::Alloc && bytes.contains(&e.new_size))
        .count() as u64
}

pub fn inventory_of<P: Inspect + ?Sized>(
    inner: &mut P,
    geometry: &[LayerContinuationGeometry],
) -> Vec<Backing> {
    let mut out = Vec::new();
    for (layer, g) in geometry.iter().enumerate() {
        if let Some(kv) = g.kv_side() {
            let row_payload = kv.kv_dim * F32_BYTES;
            for (kind, rows) in [("k_row", inner.keys(layer)), ("v_row", inner.values(layer))] {
                if !rows.is_empty() {
                    out.push(Backing {
                        kind: if kind == "k_row" {
                            "k_header"
                        } else {
                            "v_header"
                        },
                        layer,
                        index: 0,
                        ptr: rows.as_ptr() as usize,
                        bytes: alloc::live_size(rows.as_ptr() as usize),
                        payload_bytes: 0,
                    });
                }
                for (index, row) in rows.iter().enumerate() {
                    out.push(Backing {
                        kind,
                        layer,
                        index,
                        ptr: row.as_ptr() as usize,
                        bytes: Some(row.capacity() * F32_BYTES),
                        payload_bytes: row_payload,
                    });
                }
            }
            if let (Some((k, v)), Some(rows)) = (inner.matrix_ptrs(layer), inner.matrix_rows(layer))
            {
                for (kind, ptr) in [("k_matrix", k), ("v_matrix", v)] {
                    out.push(Backing {
                        kind,
                        layer,
                        index: 0,
                        ptr,
                        bytes: alloc::live_size(ptr),
                        payload_bytes: rows * row_payload,
                    });
                }
            }
        }
        if g.recurrent().is_some() {
            let state = inner.recurrent_state(layer).expect("declared recurrent");
            for index in 0..state.len() {
                let cells = state.buffer(index).cells();
                out.push(Backing {
                    kind: "recurrent",
                    layer,
                    index,
                    ptr: cells.as_ptr() as usize,
                    bytes: alloc::live_size(cells.as_ptr() as usize),
                    payload_bytes: cells.len() * F32_BYTES,
                });
            }
        }
        if let Some(latent) = g.latent_kv() {
            let rows = inner.latent_state(layer).expect("declared latent");
            for (index, row) in rows.rows().iter().enumerate() {
                out.push(Backing {
                    kind: "latent_row",
                    layer,
                    index,
                    ptr: row.as_ptr() as usize,
                    bytes: Some(row.capacity() * F32_BYTES),
                    payload_bytes: latent.width * F32_BYTES,
                });
            }
        }
    }
    out
}

impl<P> Measured<P> {
    fn close_window(&self) {
        let Some((scope, window, layer, rows_held, target_bytes)) = self.open.borrow_mut().take()
        else {
            return;
        };
        let delta = scope.leave();
        let events = alloc::events(&delta);
        let target_allocs = count_allocs_of(&events, &target_bytes);
        self.intervals.borrow_mut().push(IntervalRecord {
            phase: self.phase.get(),
            window,
            layer,
            rows_held,
            delta,
            target_bytes,
            target_allocs,
        });
    }

    fn open_window(
        &self,
        window: Window,
        layer: usize,
        rows_held: usize,
        target_bytes: Vec<usize>,
    ) {
        let scope = alloc::enter();
        *self.open.borrow_mut() = Some((scope, window, layer, rows_held, target_bytes));
    }
}

impl<P> Drop for Measured<P> {
    fn drop(&mut self) {
        self.close_window();
    }
}

impl<P: Inspect> ContinuationProvider for Measured<P> {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        self.close_window();
        let scope = alloc::enter();
        self.inner.prepare(layers);
        let delta = scope.leave();
        if self.geometry.is_empty() {
            self.geometry = layers
                .iter()
                .cloned()
                .map(LayerContinuationGeometry::Kv)
                .collect();
        }
        self.record(Method::Prepare, None, None, delta);
    }

    fn prepare_continuation(
        &mut self,
        layers: &[LayerContinuationGeometry],
    ) -> Result<(), ContinuationError> {
        self.close_window();
        let scope = alloc::enter();
        let result = self.inner.prepare_continuation(layers);
        let delta = scope.leave();
        self.geometry = layers.to_vec();
        self.record(Method::PrepareContinuation, None, None, delta);
        result
    }

    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        self.close_window();
        let overlay_rows = match &self.overlay {
            Some(o) if o.layer == layer => Some((key.clone(), value.clone())),
            _ => None,
        };
        let key_ptr = key.as_ptr() as usize;
        let moved_in_bytes = ((key.capacity() + value.capacity()) * F32_BYTES) as u64;
        let incoming = [key_ptr, value.as_ptr() as usize];
        let rows_before = self.inner.matrix_rows(layer);
        let scope = alloc::enter();
        self.inner.append(layer, key, value);
        let delta = scope.leave();
        let events = alloc::events(&delta);

        let row_bytes = self.row_bytes(layer);
        let stored_k = self.inner.keys(layer).last().map(|r| r.as_ptr() as usize);
        let stored_v = self.inner.values(layer).last().map(|r| r.as_ptr() as usize);
        let headers = [
            self.inner.keys(layer).as_ptr() as usize,
            self.inner.values(layer).as_ptr() as usize,
        ];
        let matrix = self.inner.matrix_ptrs(layer);
        let mut t = AppendTraffic {
            adopted: stored_k == Some(key_ptr),
            matrix_grew_by_one: self
                .inner
                .matrix_rows(layer)
                .map(|after| after == rows_before.unwrap_or(0) + 1),
            ..AppendTraffic::default()
        };
        for e in &events {
            let is_matrix = matrix.is_some_and(|(k, v)| e.new_ptr == k || e.new_ptr == v);
            let is_view = Some(e.new_ptr) == stored_k || Some(e.new_ptr) == stored_v;
            let is_header = headers.contains(&e.new_ptr);
            if e.kind == EventKind::Alloc && e.new_size == row_bytes {
                t.row_sized_allocs += 1;
            }
            match e.kind {
                EventKind::Free if incoming.contains(&e.old_ptr) => t.incoming_freed += 1,
                // A matrix reallocation frees its old block inside realloc,
                // not as a separate free; a separate free here is foreign
                // to every class above.
                EventKind::Free => t.unclassified += 1,
                EventKind::Alloc if is_view => {
                    t.duplicate_allocs += 1;
                    t.duplicate_bytes += e.new_size as u64;
                }
                EventKind::Alloc if is_matrix => t.matrix_first_alloc_bytes += e.new_size as u64,
                EventKind::ReallocMoved if is_matrix => {
                    t.matrix_moved += 1;
                    t.matrix_moved_bytes += e.old_size as u64;
                }
                EventKind::ReallocInPlace if is_matrix => t.matrix_in_place += 1,
                EventKind::Alloc | EventKind::ReallocMoved | EventKind::ReallocInPlace
                    if is_header =>
                {
                    t.header_alloc_bytes += e.new_size as u64
                }
                _ => t.unclassified += 1,
            }
        }
        if let Some((k, v)) = overlay_rows {
            let o = self.overlay.as_mut().expect("matched above");
            o.keys.push(k);
            o.values.push(v);
        }
        self.record(Method::Append, Some(layer), None, delta);
        let mut calls = self.calls.borrow_mut();
        let last = calls.last_mut().expect("just recorded");
        last.append = Some(t);
        last.moved_in_bytes = moved_in_bytes;
    }

    fn keys(&self, layer: usize) -> &[Vec<f32>] {
        self.close_window();
        let scope = alloc::enter();
        let rows = self.inner.keys(layer);
        let delta = scope.leave();
        self.record(Method::Keys, Some(layer), Some(rows.len()), delta);
        if self.conv_qkv.get(layer).copied().unwrap_or(false) {
            self.open_window(
                Window::AfterKeys,
                layer,
                rows.len(),
                vec![self.row_bytes(layer)],
            );
        }
        match &self.overlay {
            Some(o) if o.layer == layer => &o.keys,
            _ => rows,
        }
    }

    fn values(&self, layer: usize) -> &[Vec<f32>] {
        self.close_window();
        let scope = alloc::enter();
        let rows = self.inner.values(layer);
        let delta = scope.leave();
        self.record(Method::Values, Some(layer), Some(rows.len()), delta);
        if self.conv_qkv.get(layer).copied().unwrap_or(false) {
            self.open_window(
                Window::AfterValues,
                layer,
                rows.len(),
                vec![self.row_bytes(layer)],
            );
        }
        match &self.overlay {
            Some(o) if o.layer == layer => &o.values,
            _ => rows,
        }
    }

    fn position(&self) -> usize {
        self.close_window();
        let scope = alloc::enter();
        let p = self.inner.position();
        let delta = scope.leave();
        self.record(Method::Position, None, None, delta);
        p
    }

    fn set_position(&mut self, position: usize) {
        self.close_window();
        let scope = alloc::enter();
        self.inner.set_position(position);
        let delta = scope.leave();
        self.record(Method::SetPosition, None, None, delta);
        let triggered = self
            .sink
            .as_ref()
            .is_some_and(|(trigger, _)| trigger.swap(false, Ordering::SeqCst));
        if triggered {
            let inventory = inventory_of(&mut self.inner, &self.geometry);
            let (_, sink) = self.sink.as_ref().expect("checked");
            *sink.lock().expect("sink") = Some(inventory);
        }
    }

    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        self.close_window();
        let scope = alloc::enter();
        let ok = self.inner.recurrent_state(layer).is_ok();
        let delta = scope.leave();
        self.record(Method::Recurrent, Some(layer), None, delta);
        if ok && self.watch_recurrent {
            if let Some(Some(target)) = self.recurrent_target.get(layer) {
                let held = self.inner.keys_len_or_zero(layer);
                self.open_window(Window::AfterRecurrent, layer, held, target.clone());
            }
        }
        self.inner.recurrent_state(layer)
    }

    fn latent_state(&mut self, layer: usize) -> Result<&mut LatentKvRows, ContinuationError> {
        self.close_window();
        let scope = alloc::enter();
        let held = self.inner.latent_state(layer).map(|r| r.len()).ok();
        let delta = scope.leave();
        self.record(Method::Latent, Some(layer), held, delta);
        self.inner.latent_state(layer)
    }
}

/// Rows held by a layer that may not keep KV rows at all.
trait RowsHeld {
    fn keys_len_or_zero(&self, layer: usize) -> usize;
}

impl<P: Inspect> RowsHeld for P {
    fn keys_len_or_zero(&self, layer: usize) -> usize {
        // `keys` on a pure-recurrent layer of a row store is an empty
        // slice; on the canonical store its view list is empty too.
        self.keys(layer).len()
    }
}
