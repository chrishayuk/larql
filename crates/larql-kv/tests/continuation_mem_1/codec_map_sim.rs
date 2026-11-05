//! CONTINUATION-CODEC-MAP-1 reconnaissance: a SIMULATED partial codec.
//!
//! `MappedCodec` holds codec/v1's retention (the plan's
//! `HistoryRange::required_start`), encodes every appended K and V head
//! ONCE with codec/v1's own TurboQuant at the same bits, and keeps the
//! exact row beside the decoded one. At `prepare_layer` it lays out one
//! contiguous scratch, as codec/v1 does, taking each row's K and V from
//! the decoded or the exact copy according to a `CompressMap` (layers, K/V,
//! row age). With every row compressed it must be behaviourally identical
//! to codec/v1 — SIM-0, established by `Recorder` before any map is read.
//! A test instrument, not a provider: it holds f32 copies by design.

use std::cell::Cell;

use larql_kv::engines::turbo_quant::TurboQuant;
use larql_kv::CodecKvState;
use larql_vindex::format::vindex3::opplan::exec::continuation::{
    LatentKvRows, LayerContinuationGeometry, RecurrentState,
};
use larql_vindex::format::vindex3::opplan::exec::kv::{
    ContinuationError, ContinuationProvider, HistoryRange, LayerKvGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::kv_view::KvView;

use super::measured::{CodeList, Inspect};

const NAME: &str = "MappedCodec";
/// Depth groups the map grammar can name (`q1`..`q4`).
pub const DEPTH_GROUPS: usize = 4;

/// Which layers a map compresses.
#[derive(Clone, Debug, PartialEq)]
pub enum Layers {
    All,
    /// Layers whose plan retention is a trailing window.
    Sliding,
    /// Layers that keep every position.
    Full,
    /// Depth group `g` of DEPTH_GROUPS (0-based), by layer index.
    Depth(usize),
    Only(Vec<usize>),
}

/// Which rows, by age (newest row = age 0 at the read), a map compresses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Age {
    All,
    /// Rows of age ≥ w — the most recent w stay exact.
    OlderThan(usize),
    /// Rows of age < w — older rows stay exact (the inverse control).
    NewerThan(usize),
}

impl Age {
    fn compresses(self, age: usize) -> bool {
        match self {
            Age::All => true,
            Age::OlderThan(w) => age >= w,
            Age::NewerThan(w) => age < w,
        }
    }
}

/// One selection: these layers' K and/or V rows of this age are compressed.
#[derive(Clone, Debug, PartialEq)]
pub struct Clause {
    pub layers: Layers,
    pub k: bool,
    pub v: bool,
    pub age: Age,
}

impl Clause {
    pub fn all() -> Self {
        Self {
            layers: Layers::All,
            k: true,
            v: true,
            age: Age::All,
        }
    }
}

/// A named union of clauses: a row's K (or V) is compressed if ANY clause
/// selects its layer, that tensor and its age. A union expresses a
/// protection map — e.g. every V, and K outside a protected set.
#[derive(Clone, Debug, PartialEq)]
pub struct CompressMap {
    pub name: String,
    pub clauses: Vec<Clause>,
}

impl CompressMap {
    pub fn all() -> Self {
        Self::single("all", Clause::all())
    }

    pub fn single(name: &str, clause: Clause) -> Self {
        Self {
            name: name.into(),
            clauses: vec![clause],
        }
    }

    /// `name=<clause>|<clause>…`, each clause `<layers>:<tensors>:<age>`:
    /// layers `all|sliding|full|q1..q4|l<i>+l<j>…`, tensors `k|v|kv`, age
    /// `all|older<w>|newer<w>`.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let (name, body) = spec.split_once('=').ok_or(format!("{spec}: no `name=`"))?;
        let clauses = body
            .split('|')
            .map(|c| Clause::parse(spec, c))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            name: name.to_string(),
            clauses,
        })
    }

    /// Indices of the clauses that select `layer`.
    fn selecting(&self, layer: usize, layers: usize, g: &LayerKvGeometry) -> Vec<usize> {
        (0..self.clauses.len())
            .filter(|&c| self.clauses[c].selects(layer, layers, g))
            .collect()
    }

    /// Whether `tensor` (K when `key`) of a row of `age` is compressed,
    /// given the clauses selecting its layer.
    fn compresses(&self, clauses: &[usize], age: usize, key: bool) -> bool {
        clauses.iter().any(|&c| {
            let cl = &self.clauses[c];
            (if key { cl.k } else { cl.v }) && cl.age.compresses(age)
        })
    }
}

impl Clause {
    fn parse(spec: &str, clause: &str) -> Result<Self, String> {
        let parts: Vec<&str> = clause.split(':').collect();
        let [layers, tensors, age] = parts[..] else {
            return Err(format!("{spec}: want <layers>:<tensors>:<age>"));
        };
        let layers = match layers {
            "all" => Layers::All,
            "sliding" => Layers::Sliding,
            "full" => Layers::Full,
            q if q.starts_with('q') => {
                let g: usize = q[1..]
                    .parse()
                    .map_err(|_| format!("{spec}: bad depth {q}"))?;
                if !(1..=DEPTH_GROUPS).contains(&g) {
                    return Err(format!(
                        "{spec}: depth group {g} outside 1..={DEPTH_GROUPS}"
                    ));
                }
                Layers::Depth(g - 1)
            }
            list => Layers::Only(
                list.split('+')
                    .map(|l| l.strip_prefix('l').and_then(|n| n.parse().ok()))
                    .collect::<Option<Vec<usize>>>()
                    .ok_or(format!("{spec}: bad layer list {list}"))?,
            ),
        };
        let (k, v) = match tensors {
            "k" => (true, false),
            "v" => (false, true),
            "kv" => (true, true),
            t => return Err(format!("{spec}: bad tensors {t}")),
        };
        let number = |s: &str| {
            s.parse::<usize>()
                .map_err(|_| format!("{spec}: bad age {age}"))
        };
        let age = match age {
            "all" => Age::All,
            a if a.starts_with("older") => Age::OlderThan(number(&a[5..])?),
            a if a.starts_with("newer") => Age::NewerThan(number(&a[5..])?),
            a => return Err(format!("{spec}: bad age {a}")),
        };
        Ok(Self { layers, k, v, age })
    }

    fn selects(&self, layer: usize, layers: usize, g: &LayerKvGeometry) -> bool {
        match &self.layers {
            Layers::All => true,
            Layers::Sliding => matches!(g.history, HistoryRange::Trailing(_)),
            Layers::Full => matches!(g.history, HistoryRange::Full),
            Layers::Depth(q) => layer * DEPTH_GROUPS / layers == *q,
            Layers::Only(list) => list.contains(&layer),
        }
    }
}

struct Layer {
    geometry: LayerKvGeometry,
    /// The clauses that select this layer.
    clauses: Vec<usize>,
    base: usize,
    exact_k: Vec<Vec<f32>>,
    exact_v: Vec<Vec<f32>>,
    coded_k: Vec<Vec<f32>>,
    coded_v: Vec<Vec<f32>>,
}

pub struct MappedCodec {
    codec: TurboQuant,
    map: CompressMap,
    layers: Vec<Layer>,
    position: usize,
    scratch_layer: Option<usize>,
    scratch_base: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
    ws_f32: Vec<f32>,
    ws_u8: Vec<u8>,
    encoded: Vec<u8>,
}

impl MappedCodec {
    pub fn new(bits: u8, map: CompressMap) -> Self {
        Self {
            codec: TurboQuant::new(bits),
            map,
            layers: Vec::new(),
            position: 0,
            scratch_layer: None,
            scratch_base: 0,
            keys: Vec::new(),
            values: Vec::new(),
            ws_f32: Vec::new(),
            ws_u8: Vec::new(),
            encoded: Vec::new(),
        }
    }

    /// codec/v1's per-head encode then decode of one row.
    fn round_trip(&mut self, row: &[f32], head_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0; row.len()];
        self.ws_f32.resize(head_dim, 0.0);
        for (head, dst) in row
            .chunks_exact(head_dim)
            .zip(out.chunks_exact_mut(head_dim))
        {
            self.encoded.clear();
            self.codec.encode_vector_into(
                head,
                &mut self.encoded,
                &mut self.ws_f32,
                &mut self.ws_u8,
            );
            self.codec
                .decode_block_into(&self.encoded, dst, &mut self.ws_u8);
        }
        out
    }

    /// Whether the map compresses `layer` at all.
    pub fn selected(&self, layer: usize) -> bool {
        !self.layers[layer].clauses.is_empty()
    }

    /// The first held position and one past the last, per layer.
    pub fn retained(&self, layer: usize) -> (usize, usize) {
        let l = &self.layers[layer];
        (l.base, l.base + l.exact_k.len())
    }

    /// Of the K and V row-halves held now, the fraction the map keeps
    /// EXACT at a read of the newest position — the protected share of a
    /// mixed representation.
    pub fn exact_fraction(&self) -> f64 {
        let (mut exact, mut total) = (0usize, 0usize);
        for l in &self.layers {
            let n = l.exact_k.len();
            for i in 0..n {
                for key in [true, false] {
                    total += l.geometry.kv_dim;
                    if !self.map.compresses(&l.clauses, n - 1 - i, key) {
                        exact += l.geometry.kv_dim;
                    }
                }
            }
        }
        exact as f64 / total.max(1) as f64
    }
}

impl Inspect for MappedCodec {
    fn matrix_ptrs(&self, _: usize) -> Option<(usize, usize)> {
        None
    }
    fn matrix_rows(&self, _: usize) -> Option<usize> {
        None
    }
    fn code_list(&self, _: usize) -> Option<CodeList> {
        None
    }
    fn code_rows(&self, _: usize, _: std::ops::Range<usize>) -> Vec<usize> {
        Vec::new()
    }
    fn scratch_ptrs(&self) -> Option<[usize; 2]> {
        None
    }
}

impl ContinuationProvider for MappedCodec {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        if self.layers.is_empty() {
            let n = layers.len();
            self.layers = layers
                .iter()
                .enumerate()
                .map(|(i, g)| Layer {
                    geometry: *g,
                    clauses: self.map.selecting(i, n, g),
                    base: 0,
                    exact_k: Vec::new(),
                    exact_v: Vec::new(),
                    coded_k: Vec::new(),
                    coded_v: Vec::new(),
                })
                .collect();
        }
    }

    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        let head_dim = self.layers[layer].geometry.head_dim;
        let (ck, cv) = (
            self.round_trip(&key, head_dim),
            self.round_trip(&value, head_dim),
        );
        let l = &mut self.layers[layer];
        let position = l.base + l.exact_k.len();
        l.exact_k.push(key);
        l.exact_v.push(value);
        l.coded_k.push(ck);
        l.coded_v.push(cv);
        let floor = l.geometry.history.required_start(position);
        let drop = floor.saturating_sub(l.base).min(l.exact_k.len());
        for rows in [
            &mut l.exact_k,
            &mut l.exact_v,
            &mut l.coded_k,
            &mut l.coded_v,
        ] {
            rows.drain(..drop);
        }
        l.base += drop;
    }

    fn prepare_layer(&mut self, layer: usize) {
        let l = &self.layers[layer];
        let (kv_dim, n) = (l.geometry.kv_dim, l.exact_k.len());
        self.keys.clear();
        self.values.clear();
        for i in 0..n {
            let age = n - 1 - i;
            let pick = |key: bool, coded: &Vec<Vec<f32>>, exact: &Vec<Vec<f32>>| {
                if self.map.compresses(&l.clauses, age, key) {
                    coded[i].clone()
                } else {
                    exact[i].clone()
                }
            };
            self.keys.extend(pick(true, &l.coded_k, &l.exact_k));
            self.values.extend(pick(false, &l.coded_v, &l.exact_v));
        }
        debug_assert_eq!(self.keys.len(), n * kv_dim);
        self.scratch_layer = Some(layer);
        self.scratch_base = l.base;
    }

    fn rows(&self, layer: usize) -> KvView<'_> {
        if self.scratch_layer != Some(layer) {
            return KvView::empty();
        }
        let kv_dim = self.layers[layer].geometry.kv_dim;
        KvView::contiguous(self.scratch_base, kv_dim, &self.keys, &self.values)
            .expect("the scratch holds whole rows of this layer's width")
    }

    fn position(&self) -> usize {
        self.position
    }

    fn set_position(&mut self, position: usize) {
        self.position = position;
    }

    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        Err(ContinuationError::RecurrentUnsupported {
            provider: NAME,
            layer,
        })
    }

    fn latent_state(&mut self, layer: usize) -> Result<&mut LatentKvRows, ContinuationError> {
        Err(ContinuationError::LatentUnsupported {
            provider: NAME,
            layer,
        })
    }
}

/// A provider whose retained range the recorder can read.
pub trait Retained {
    fn retained(&self, layer: usize) -> (usize, usize);
}

impl Retained for MappedCodec {
    fn retained(&self, layer: usize) -> (usize, usize) {
        MappedCodec::retained(self, layer)
    }
}

impl Retained for CodecKvState {
    fn retained(&self, layer: usize) -> (usize, usize) {
        let base = self.rows_base(layer);
        (base, base + self.encoded_rows(layer).len())
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Folds every provider event — append (with the retained range after
/// it), prepare_layer, and each rows() view's range — into an FNV-1a hash
/// and a count, allocating nothing, so two providers' call-and-retention
/// sequences can be compared exactly (SIM-0).
pub struct Recorder<P> {
    pub inner: P,
    hash: Cell<u64>,
    events: Cell<u64>,
}

impl<P> Recorder<P> {
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            hash: Cell::new(FNV_OFFSET),
            events: Cell::new(0),
        }
    }

    /// (hash, event count) of everything seen so far.
    pub fn trace(&self) -> (u64, u64) {
        (self.hash.get(), self.events.get())
    }

    fn fold(&self, words: [u64; 4]) {
        let mut h = self.hash.get();
        for w in words {
            for b in w.to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
            }
        }
        self.hash.set(h);
        self.events.set(self.events.get() + 1);
    }
}

impl<P: Inspect + Retained> Inspect for Recorder<P> {
    fn matrix_ptrs(&self, l: usize) -> Option<(usize, usize)> {
        self.inner.matrix_ptrs(l)
    }
    fn matrix_rows(&self, l: usize) -> Option<usize> {
        self.inner.matrix_rows(l)
    }
    fn code_list(&self, l: usize) -> Option<CodeList> {
        self.inner.code_list(l)
    }
    fn code_rows(&self, l: usize, r: std::ops::Range<usize>) -> Vec<usize> {
        self.inner.code_rows(l, r)
    }
    fn scratch_ptrs(&self) -> Option<[usize; 2]> {
        self.inner.scratch_ptrs()
    }
    fn mixed_layout(&self, l: usize) -> Option<super::measured::MixedLayout> {
        self.inner.mixed_layout(l)
    }
}

impl<P: ContinuationProvider + Retained> ContinuationProvider for Recorder<P> {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        self.inner.prepare(layers)
    }
    fn prepare_continuation(
        &mut self,
        layers: &[LayerContinuationGeometry],
    ) -> Result<(), ContinuationError> {
        self.inner.prepare_continuation(layers)
    }
    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        self.inner.append(layer, key, value);
        let (base, end) = self.inner.retained(layer);
        self.fold([1, layer as u64, base as u64, end as u64]);
    }
    fn prepare_layer(&mut self, layer: usize) {
        self.inner.prepare_layer(layer);
        self.fold([2, layer as u64, 0, 0]);
    }
    fn rows(&self, layer: usize) -> KvView<'_> {
        let view = self.inner.rows(layer);
        self.fold([3, layer as u64, view.base() as u64, view.end() as u64]);
        view
    }
    fn position(&self) -> usize {
        self.inner.position()
    }
    fn set_position(&mut self, position: usize) {
        self.inner.set_position(position)
    }
    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        self.inner.recurrent_state(layer)
    }
    fn latent_state(&mut self, layer: usize) -> Result<&mut LatentKvRows, ContinuationError> {
        self.inner.latent_state(layer)
    }
}
