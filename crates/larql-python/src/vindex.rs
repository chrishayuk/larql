//! Python bindings for the vindex — the queryable model format.
//!
//! Exposes VectorIndex + embeddings + tokenizer as a single Python object
//! with numpy array returns for gate vectors, embeddings, and KNN results.
//!
//! Two access patterns:
//! - Direct API: gate_vector(), embed(), gate_knn() — raw numpy arrays
//! - High-level: describe(), entity_knn(), insert() — string in, results out

use std::collections::HashMap;

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use larql_vindex::patch::knn_store::KnnStore;
use larql_vindex::{
    format::filenames::KNN_STORE_BIN, load_vindex_config, load_vindex_embeddings,
    load_vindex_tokenizer, tokenizers, FeatureMeta, SilentLoadCallbacks, VectorIndex, VindexConfig,
    WalkHit,
};

use larql_lql::relations::RelationClassifier;

// ── Content token filter (matches LQL executor logic) ──

fn is_readable_token(tok: &str) -> bool {
    let tok = tok.trim();
    if tok.is_empty() || tok.len() > 30 {
        return false;
    }
    let readable = tok
        .chars()
        .filter(|c| {
            c.is_ascii_alphanumeric()
                || *c == ' '
                || *c == '-'
                || *c == '\''
                || *c == '.'
                || *c == ','
        })
        .count();
    let total = tok.chars().count();
    readable * 2 >= total && total > 0
}

fn is_content_token(tok: &str) -> bool {
    let tok = tok.trim();
    if !is_readable_token(tok) {
        return false;
    }
    let chars: Vec<char> = tok.chars().collect();
    if chars.len() < 3 || chars.len() > 25 {
        return false;
    }
    let alpha = chars.iter().filter(|c| c.is_ascii_alphabetic()).count();
    if alpha < chars.len() * 2 / 3 {
        return false;
    }
    for w in chars.windows(2) {
        if w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase() {
            return false;
        }
    }
    if !chars.iter().any(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    let lower = tok.to_lowercase();
    !matches!(
        lower.as_str(),
        "the"
            | "and"
            | "for"
            | "but"
            | "not"
            | "you"
            | "all"
            | "can"
            | "her"
            | "was"
            | "one"
            | "our"
            | "out"
            | "are"
            | "has"
            | "his"
            | "how"
            | "its"
            | "may"
            | "new"
            | "now"
            | "old"
            | "see"
            | "way"
            | "who"
            | "did"
            | "get"
            | "let"
            | "say"
            | "she"
            | "too"
            | "use"
            | "from"
            | "have"
            | "been"
            | "will"
            | "with"
            | "this"
            | "that"
            | "they"
            | "were"
            | "some"
            | "them"
            | "than"
            | "when"
            | "what"
            | "your"
            | "each"
            | "make"
            | "like"
            | "just"
            | "over"
            | "such"
            | "take"
            | "also"
            | "into"
            | "only"
            | "very"
            | "more"
            | "does"
            | "most"
            | "about"
            | "which"
            | "their"
            | "would"
            | "there"
            | "could"
            | "other"
            | "after"
            | "being"
            | "where"
            | "these"
            | "those"
            | "first"
            | "should"
            | "because"
            | "through"
            | "before"
            | "par"
            | "aux"
            | "che"
            | "del"
    )
}

// ── PyDescribeEdge ──

#[pyclass(name = "DescribeEdge")]
#[derive(Clone)]
pub struct PyDescribeEdge {
    #[pyo3(get)]
    pub relation: Option<String>,
    #[pyo3(get)]
    pub source: String,
    #[pyo3(get)]
    pub target: String,
    #[pyo3(get)]
    pub gate_score: f32,
    #[pyo3(get)]
    pub layer: usize,
    #[pyo3(get)]
    pub feature: usize,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub also: Vec<String>,
}

#[pymethods]
impl PyDescribeEdge {
    fn __repr__(&self) -> String {
        let rel = self.relation.as_deref().unwrap_or("?");
        format!(
            "DescribeEdge(relation='{}', target='{}', score={:.1}, layer={}, source='{}')",
            rel, self.target, self.gate_score, self.layer, self.source
        )
    }
}

// ── PyRelation ──

#[pyclass(name = "Relation")]
#[derive(Clone)]
pub struct PyRelation {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub cluster_id: usize,
    #[pyo3(get)]
    pub count: usize,
    #[pyo3(get)]
    pub top_tokens: Vec<String>,
}

#[pymethods]
impl PyRelation {
    fn __repr__(&self) -> String {
        format!(
            "Relation(name='{}', count={}, cluster={})",
            self.name, self.count, self.cluster_id
        )
    }
}

// ── PyFeatureMeta ──

#[pyclass(name = "FeatureMeta")]
#[derive(Clone)]
pub struct PyFeatureMeta {
    inner: FeatureMeta,
}

#[pymethods]
impl PyFeatureMeta {
    #[getter]
    fn top_token(&self) -> &str {
        &self.inner.top_token
    }

    #[getter]
    fn top_token_id(&self) -> u32 {
        self.inner.top_token_id
    }

    #[getter]
    fn c_score(&self) -> f32 {
        self.inner.c_score
    }

    #[getter]
    fn top_k<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut result = Vec::new();
        for entry in &self.inner.top_k {
            let dict = PyDict::new(py);
            dict.set_item("token", &entry.token)?;
            dict.set_item("token_id", entry.token_id)?;
            dict.set_item("logit", entry.logit)?;
            result.push(dict);
        }
        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!(
            "FeatureMeta(token='{}', id={}, c={:.4})",
            self.inner.top_token, self.inner.top_token_id, self.inner.c_score
        )
    }
}

// ── PyWalkHit ──

#[pyclass(name = "WalkHit")]
#[derive(Clone)]
pub struct PyWalkHit {
    inner_layer: usize,
    inner_feature: usize,
    inner_gate_score: f32,
    inner_meta: FeatureMeta,
}

#[pymethods]
impl PyWalkHit {
    #[getter]
    fn layer(&self) -> usize {
        self.inner_layer
    }

    #[getter]
    fn feature(&self) -> usize {
        self.inner_feature
    }

    #[getter]
    fn gate_score(&self) -> f32 {
        self.inner_gate_score
    }

    #[getter]
    fn meta(&self) -> PyFeatureMeta {
        PyFeatureMeta {
            inner: self.inner_meta.clone(),
        }
    }

    #[getter]
    fn top_token(&self) -> &str {
        &self.inner_meta.top_token
    }

    #[getter]
    fn target(&self) -> &str {
        &self.inner_meta.top_token
    }

    fn __repr__(&self) -> String {
        format!(
            "WalkHit(L{}:F{} score={:.4} token='{}')",
            self.inner_layer, self.inner_feature, self.inner_gate_score, self.inner_meta.top_token
        )
    }
}

impl From<WalkHit> for PyWalkHit {
    fn from(h: WalkHit) -> Self {
        Self {
            inner_layer: h.layer,
            inner_feature: h.feature,
            inner_gate_score: h.gate_score,
            inner_meta: h.meta,
        }
    }
}

// ── PyVindex ──

#[pyclass(name = "Vindex", unsendable)]
pub struct PyVindex {
    pub(crate) index: VectorIndex,
    pub(crate) embeddings: ndarray::Array2<f32>,
    pub(crate) embed_scale: f32,
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) config: VindexConfig,
    pub(crate) path: String,
    pub(crate) classifier: Option<RelationClassifier>,
    /// Arch-B retrieval-override store. Loaded from `knn_store.bin` at
    /// open time if present. `infer()` captures residuals and consults
    /// this store before returning the raw model prediction; a stored
    /// key with `cos > KNN_COSINE_THRESHOLD` overrides the top-1
    /// prediction with the stored target token. Matches the LQL INFER
    /// query path (`executor/query/infer.rs`).
    pub(crate) knn_store: Option<KnnStore>,
    /// Lazy-loaded mmap'd weights for infer(). Created on first call, reused after.
    pub(crate) walk_model: std::cell::RefCell<Option<crate::walk::InferState>>,
}

impl PyVindex {
    /// Load a vindex from a directory path (Rust-callable).
    pub fn open(path: &str) -> PyResult<Self> {
        let dir = std::path::Path::new(path);

        let config = load_vindex_config(dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let mut callbacks = SilentLoadCallbacks;
        let index = VectorIndex::load_vindex(dir, &mut callbacks)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let (embeddings, embed_scale) = load_vindex_embeddings(dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let tokenizer = load_vindex_tokenizer(dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Load relation classifier (clusters + labels) if available
        let classifier = RelationClassifier::from_vindex(dir);

        // Load the arch-B KNN store if the compiled vindex bundled one.
        let knn_path = dir.join(KNN_STORE_BIN);
        let knn_store = if knn_path.exists() {
            match KnnStore::load(&knn_path) {
                Ok(store) => Some(store),
                Err(e) => {
                    eprintln!("warning: failed to load knn_store.bin: {e}");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            index,
            embeddings,
            embed_scale,
            tokenizer,
            config,
            path: path.to_string(),
            classifier,
            knn_store,
            walk_model: std::cell::RefCell::new(None),
        })
    }

    /// Run a closure with a reference to the lazily-loaded walk FFN state.
    /// Loads on first call; subsequent calls reuse the mmap'd weights.
    fn with_walk_model<F, R>(&self, f: F) -> PyResult<R>
    where
        F: FnOnce(&crate::walk::InferState) -> PyResult<R>,
    {
        {
            let mut state = self.walk_model.borrow_mut();
            if state.is_none() {
                let dir = std::path::Path::new(&self.path);
                *state = Some(crate::walk::InferState::load(dir).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to load model weights: {e}"
                    ))
                })?);
            }
        }
        let state = self.walk_model.borrow();
        f(state.as_ref().unwrap())
    }

    /// Compute scaled embedding for entity text. Multi-token entities are averaged.
    fn compute_embed(&self, text: &str) -> PyResult<Array1<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let ids = encoding.get_ids();
        if ids.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Empty tokenization",
            ));
        }

        let hidden = self.config.hidden_size;
        let mut sum = Array1::<f32>::zeros(hidden);
        let mut count = 0usize;

        for &tid in ids {
            let id = tid as usize;
            if id < self.embeddings.shape()[0] {
                sum += &self.embeddings.row(id);
                count += 1;
            }
        }

        if count == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No valid token IDs",
            ));
        }

        let avg = sum / count as f32;
        Ok(avg * self.embed_scale)
    }
}

#[pymethods]
impl PyVindex {
    /// Load a vindex from a directory path.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        Self::open(path)
    }

    // ══════════════════════════════════════════════
    //  Properties
    // ══════════════════════════════════════════════

    #[getter]
    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    #[getter]
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    #[getter]
    fn model(&self) -> &str {
        &self.config.model
    }

    #[getter]
    fn family(&self) -> &str {
        &self.config.family
    }

    #[getter]
    fn is_mmap(&self) -> bool {
        self.index.is_mmap()
    }

    #[getter]
    fn total_gate_vectors(&self) -> usize {
        self.index.total_gate_vectors()
    }

    #[getter]
    fn loaded_layers(&self) -> Vec<usize> {
        self.index.loaded_layers()
    }

    #[getter]
    fn embed_scale_value(&self) -> f32 {
        self.embed_scale
    }

    /// Number of features at a layer.
    fn num_features(&self, layer: usize) -> usize {
        self.index.num_features(layer)
    }

    // ══════════════════════════════════════════════
    //  Embeddings
    // ══════════════════════════════════════════════

    /// Embed entity text as a scaled numpy array.
    /// Multi-token entities are averaged (e.g., "John Coyle" averages both tokens).
    fn embed<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let arr = self.compute_embed(text)?;
        Ok(arr.to_vec().into_pyarray(py))
    }

    /// Tokenize text and return all token IDs.
    fn tokenize(&self, text: &str) -> PyResult<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text.
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.tokenizer
            .decode(&ids, true)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Get the raw embedding for a token ID (unscaled).
    fn embedding<'py>(
        &self,
        py: Python<'py>,
        token_id: u32,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let id = token_id as usize;
        if id >= self.embeddings.shape()[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Token ID {} out of range",
                token_id
            )));
        }
        Ok(self.embeddings.row(id).to_vec().into_pyarray(py))
    }

    /// Get the full embedding matrix as numpy (vocab_size, hidden_size).
    fn embedding_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (rows, cols) = (self.embeddings.shape()[0], self.embeddings.shape()[1]);
        let data = if let Some(slice) = self.embeddings.as_slice() {
            slice.to_vec()
        } else {
            let mut data = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                data.extend(self.embeddings.row(r).iter());
            }
            data
        };
        let arr = ndarray::Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    // ══════════════════════════════════════════════
    //  Gate vectors
    // ══════════════════════════════════════════════

    /// Get a single gate vector as numpy array (hidden_size,).
    fn gate_vector<'py>(
        &self,
        py: Python<'py>,
        layer: usize,
        feature: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.index
            .gate_vector(layer, feature)
            .map(|v| v.into_pyarray(py))
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "No gate vector at L{}:F{}",
                    layer, feature
                ))
            })
    }

    /// Get all gate vectors at a layer as numpy (num_features, hidden_size).
    fn gate_vectors<'py>(
        &self,
        py: Python<'py>,
        layer: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (data, rows, cols) = self.index.gate_vectors_flat(layer).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("No gate vectors at layer {}", layer))
        })?;
        let arr = ndarray::Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    // ══════════════════════════════════════════════
    //  KNN & Walk
    // ══════════════════════════════════════════════

    /// Gate KNN: find top-K features at a layer by dot product with a query vector.
    /// Returns list of (feature_index, score) tuples.
    #[pyo3(signature = (layer, query_vector, top_k=10))]
    fn gate_knn(&self, layer: usize, query_vector: Vec<f32>, top_k: usize) -> Vec<(usize, f32)> {
        let arr = Array1::from_vec(query_vector);
        self.index.gate_knn(layer, &arr, top_k)
    }

    /// Walk: gate KNN across multiple layers with a raw residual vector.
    /// Returns list of WalkHit objects.
    #[pyo3(signature = (residual, layers=None, top_k=5))]
    fn walk(&self, residual: Vec<f32>, layers: Option<Vec<usize>>, top_k: usize) -> Vec<PyWalkHit> {
        let arr = Array1::from_vec(residual);
        let layer_list = layers.unwrap_or_else(|| self.index.loaded_layers());
        let trace = self.index.walk(&arr, &layer_list, top_k);
        trace
            .layers
            .into_iter()
            .flat_map(|(_, hits)| hits.into_iter().map(PyWalkHit::from))
            .collect()
    }

    /// Convenience: embed entity text and walk across layers.
    /// Like walk() but takes a string instead of a raw vector.
    #[pyo3(signature = (entity, layers=None, top_k=5))]
    fn entity_walk(
        &self,
        entity: &str,
        layers: Option<Vec<usize>>,
        top_k: usize,
    ) -> PyResult<Vec<PyWalkHit>> {
        let arr = self.compute_embed(entity)?;
        let layer_list = layers.unwrap_or_else(|| self.index.loaded_layers());
        let trace = self.index.walk(&arr, &layer_list, top_k);
        Ok(trace
            .layers
            .into_iter()
            .flat_map(|(_, hits)| hits.into_iter().map(PyWalkHit::from))
            .collect())
    }

    /// Convenience: embed entity and do gate KNN at a layer.
    #[pyo3(signature = (entity, layer, top_k=10))]
    fn entity_knn(&self, entity: &str, layer: usize, top_k: usize) -> PyResult<Vec<(usize, f32)>> {
        let arr = self.compute_embed(entity)?;
        Ok(self.index.gate_knn(layer, &arr, top_k))
    }

    // ══════════════════════════════════════════════
    //  Feature metadata
    // ══════════════════════════════════════════════

    /// Look up metadata for a specific feature. Returns FeatureMeta or None.
    fn feature_meta(&self, layer: usize, feature: usize) -> Option<PyFeatureMeta> {
        self.index
            .feature_meta(layer, feature)
            .map(|m| PyFeatureMeta { inner: m })
    }

    /// Get feature metadata as a dict (for quick inspection in notebooks).
    fn feature<'py>(
        &self,
        py: Python<'py>,
        layer: usize,
        feature: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let meta = match self.index.feature_meta(layer, feature) {
            Some(m) => m,
            None => return Ok(None),
        };
        let dict = PyDict::new(py);
        dict.set_item("layer", layer)?;
        dict.set_item("feature", feature)?;
        dict.set_item("top_token", &meta.top_token)?;
        dict.set_item("top_token_id", meta.top_token_id)?;
        dict.set_item("c_score", meta.c_score)?;
        let top_k: Vec<(&str, u32, f32)> = meta
            .top_k
            .iter()
            .map(|t| (t.token.as_str(), t.token_id, t.logit))
            .collect();
        dict.set_item("top_k", top_k)?;
        Ok(Some(dict))
    }

    /// Get the relation label for a feature (probe or cluster-assigned).
    fn feature_label(&self, layer: usize, feature: usize) -> Option<String> {
        self.classifier
            .as_ref()?
            .label_for_feature(layer, feature)
            .map(|s| s.to_string())
    }

    // ══════════════════════════════════════════════
    //  DESCRIBE — knowledge edge discovery
    // ══════════════════════════════════════════════

    /// Describe an entity: find all knowledge edges.
    ///
    /// Returns a list of DescribeEdge objects with relation labels,
    /// targets, gate scores, layer info, and secondary tokens.
    ///
    /// Args:
    ///     entity: Entity name ("France", "Einstein")
    ///     band: "knowledge" (default), "syntax", "output", or "all"
    ///     verbose: Include cluster labels (not just probe-confirmed)
    #[pyo3(signature = (entity, band="knowledge", verbose=false))]
    fn describe(&self, entity: &str, band: &str, verbose: bool) -> PyResult<Vec<PyDescribeEdge>> {
        let query = self.compute_embed(entity)?;

        // Determine which layers to scan
        let (start, end) = match band {
            "syntax" => {
                if let Some(ref b) = self.config.layer_bands {
                    (b.syntax.0, b.syntax.1)
                } else {
                    (0, self.config.num_layers / 3)
                }
            }
            "output" => {
                if let Some(ref b) = self.config.layer_bands {
                    (b.output.0, b.output.1)
                } else {
                    (self.config.num_layers * 5 / 6, self.config.num_layers - 1)
                }
            }
            "all" => (0, self.config.num_layers - 1),
            _ => {
                // "knowledge" default
                if let Some(ref b) = self.config.layer_bands {
                    (b.knowledge.0, b.knowledge.1)
                } else {
                    (self.config.num_layers / 3, self.config.num_layers * 5 / 6)
                }
            }
        };

        let scan_layers: Vec<usize> = (start..=end).collect();
        let trace = self.index.walk(&query, &scan_layers, 20);

        // Collect edges: group by token, track best score/layer
        struct EdgeAccum {
            target: String,
            best_score: f32,
            best_layer: usize,
            best_feature: usize,
            c_score: f32,
            also: Vec<String>,
        }

        let mut edge_map: HashMap<String, EdgeAccum> = HashMap::new();

        for (_, hits) in &trace.layers {
            for hit in hits {
                let tok = hit.meta.top_token.trim().to_string();
                if !is_content_token(&tok) {
                    continue;
                }
                if hit.gate_score.abs() < 5.0 {
                    continue;
                }

                let key = tok.to_lowercase();
                let entry = edge_map.entry(key).or_insert_with(|| EdgeAccum {
                    target: tok.clone(),
                    best_score: 0.0,
                    best_layer: hit.layer,
                    best_feature: hit.feature,
                    c_score: hit.meta.c_score,
                    also: Vec::new(),
                });

                if hit.gate_score.abs() > entry.best_score.abs() {
                    entry.best_score = hit.gate_score;
                    entry.best_layer = hit.layer;
                    entry.best_feature = hit.feature;
                    entry.c_score = hit.meta.c_score;
                }

                // Collect secondary tokens
                for tk in &hit.meta.top_k {
                    let sec = tk.token.trim().to_string();
                    if is_content_token(&sec)
                        && sec.to_lowercase() != entry.target.to_lowercase()
                        && !entry.also.contains(&sec)
                        && entry.also.len() < 3
                    {
                        entry.also.push(sec);
                    }
                }
            }
        }

        // Convert to edges with relation labels
        let mut edges: Vec<PyDescribeEdge> = Vec::new();
        for acc in edge_map.values() {
            let (relation, source) = if let Some(ref rc) = self.classifier {
                if let Some(label) = rc.label_for_feature(acc.best_layer, acc.best_feature) {
                    let src = if rc.is_probe_label(acc.best_layer, acc.best_feature) {
                        "probe"
                    } else if verbose {
                        "cluster"
                    } else {
                        // Skip cluster labels when not verbose
                        "cluster"
                    };
                    (Some(label.to_string()), src.to_string())
                } else {
                    (None, "none".to_string())
                }
            } else {
                (None, "none".to_string())
            };

            // In non-verbose mode, skip unlabelled edges with weak scores
            if !verbose && relation.is_none() && acc.best_score.abs() < 20.0 {
                continue;
            }

            edges.push(PyDescribeEdge {
                relation,
                source,
                target: acc.target.clone(),
                gate_score: acc.best_score,
                layer: acc.best_layer,
                feature: acc.best_feature,
                confidence: acc.c_score,
                also: acc.also.clone(),
            });
        }

        // Sort by gate score descending
        edges.sort_by(|a, b| b.gate_score.abs().partial_cmp(&a.gate_score.abs()).unwrap());
        Ok(edges)
    }

    // ══════════════════════════════════════════════
    //  Relations & Clusters
    // ══════════════════════════════════════════════

    /// List all known relation types with counts and cluster info.
    fn relations(&self) -> Vec<PyRelation> {
        let rc = match &self.classifier {
            Some(rc) if rc.has_clusters() => rc,
            _ => return Vec::new(),
        };

        let mut rels = Vec::new();
        for i in 0..rc.num_clusters() {
            if let Some((label, count, tops)) = rc.cluster_info(i) {
                // Skip garbage labels
                if label.contains('/') && label.len() > 20 {
                    continue;
                }
                rels.push(PyRelation {
                    name: label.to_string(),
                    cluster_id: i,
                    count,
                    top_tokens: tops.to_vec(),
                });
            }
        }

        rels.sort_by_key(|r| std::cmp::Reverse(r.count));
        rels
    }

    /// Get the cluster centre vector for a relation type as numpy array.
    /// Returns None if the relation is not found.
    fn cluster_centre<'py>(
        &self,
        py: Python<'py>,
        relation: &str,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let rc = match &self.classifier {
            Some(rc) => rc,
            None => return Ok(None),
        };
        Ok(rc
            .cluster_centre_for_relation(relation)
            .map(|v| v.into_pyarray(py)))
    }

    /// Get the typical layer for a relation type.
    fn typical_layer(&self, relation: &str) -> Option<usize> {
        self.classifier
            .as_ref()?
            .typical_layer_for_relation(relation)
    }

    /// Check if entity has an edge with the given relation.
    #[pyo3(signature = (entity, relation=None))]
    fn has_edge(&self, entity: &str, relation: Option<&str>) -> PyResult<bool> {
        let edges = self.describe(entity, "knowledge", false)?;
        Ok(match relation {
            Some(r) => edges.iter().any(|e| {
                e.relation
                    .as_deref()
                    .map(|l| l.eq_ignore_ascii_case(r))
                    .unwrap_or(false)
            }),
            None => !edges.is_empty(),
        })
    }

    /// Get the target token for an entity+relation pair.
    /// Returns None if not found.
    #[pyo3(signature = (entity, relation))]
    fn get_target(&self, entity: &str, relation: &str) -> PyResult<Option<String>> {
        let edges = self.describe(entity, "knowledge", false)?;
        Ok(edges
            .iter()
            .find(|e| {
                e.relation
                    .as_deref()
                    .map(|l| l.eq_ignore_ascii_case(relation))
                    .unwrap_or(false)
            })
            .map(|e| e.target.clone()))
    }

    // ══════════════════════════════════════════════
    //  Mutation — INSERT
    // ══════════════════════════════════════════════

    /// Insert a knowledge edge: synthesise gate vector and write to index.
    ///
    /// Gate vector = entity_embed * 0.7 + cluster_centre * 0.3, normalised to
    /// match existing layer magnitudes. Returns (layer, feature).
    #[pyo3(signature = (entity, relation, target, layer=None, confidence=0.8))]
    fn insert(
        &mut self,
        entity: &str,
        relation: &str,
        target: &str,
        layer: Option<usize>,
        confidence: f32,
    ) -> PyResult<(usize, usize)> {
        let entity_embed = self.compute_embed(entity)?;

        // Determine target layer
        let target_layer = layer
            .or_else(|| {
                self.classifier
                    .as_ref()?
                    .typical_layer_for_relation(relation)
            })
            .unwrap_or_else(|| {
                if let Some(ref b) = self.config.layer_bands {
                    (b.knowledge.0 + b.knowledge.1) / 2
                } else {
                    self.config.num_layers * 3 / 5
                }
            });

        // Synthesise gate vector
        let mut gate_vec = if let Some(ref rc) = self.classifier {
            if let Some(centre) = rc.cluster_centre_for_relation(relation) {
                if centre.len() == self.config.hidden_size {
                    let centre_arr = Array1::from_vec(centre);
                    &entity_embed * 0.7 + &centre_arr * 0.3
                } else {
                    entity_embed.clone()
                }
            } else {
                entity_embed.clone()
            }
        } else {
            entity_embed.clone()
        };

        // Normalise to match layer magnitudes (sample first 100 features)
        let sample_count = self.index.num_features(target_layer).min(100);
        if sample_count > 0 {
            let mut norm_sum = 0.0f32;
            let mut norm_count = 0usize;
            for f in 0..sample_count {
                if let Some(v) = self.index.gate_vector(target_layer, f) {
                    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if n > 0.0 {
                        norm_sum += n;
                        norm_count += 1;
                    }
                }
            }
            if norm_count > 0 {
                let avg_norm = norm_sum / norm_count as f32;
                let my_norm: f32 = gate_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if my_norm > 0.0 {
                    gate_vec *= avg_norm / my_norm;
                }
            }
        }

        // Find a free feature slot
        let feature = self.index.find_free_feature(target_layer).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "No free feature slot at layer {}",
                target_layer
            ))
        })?;

        // Tokenize target for metadata
        let target_encoding = self
            .tokenizer
            .encode(target, false)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let target_ids = target_encoding.get_ids();
        let target_token_id = target_ids.first().copied().unwrap_or(0);

        let meta = FeatureMeta {
            top_token: target.to_string(),
            top_token_id: target_token_id,
            c_score: confidence,
            top_k: vec![larql_models::TopKEntry {
                token: target.to_string(),
                token_id: target_token_id,
                logit: confidence,
            }],
        };

        // Write to index
        self.index.set_gate_vector(target_layer, feature, &gate_vec);
        self.index.set_feature_meta(target_layer, feature, meta);

        Ok((target_layer, feature))
    }

    /// Low-level: find a free feature slot at a layer.
    fn find_free_feature(&self, layer: usize) -> Option<usize> {
        self.index.find_free_feature(layer)
    }

    /// Low-level: set a gate vector directly. For constellation insert experiments.
    fn set_gate_vector(&mut self, layer: usize, feature: usize, vector: Vec<f32>) -> PyResult<()> {
        let arr = Array1::from_vec(vector);
        self.index.set_gate_vector(layer, feature, &arr);
        Ok(())
    }

    /// Low-level: set a custom down vector override for a feature.
    /// During inference, this vector is used instead of the model's down weight row.
    fn set_down_vector(&mut self, layer: usize, feature: usize, vector: Vec<f32>) -> PyResult<()> {
        self.index.set_down_vector(layer, feature, vector);
        Ok(())
    }

    /// Low-level: set feature metadata directly.
    #[pyo3(signature = (layer, feature, top_token, c_score=0.9))]
    fn set_feature_meta(
        &mut self,
        layer: usize,
        feature: usize,
        top_token: &str,
        c_score: f32,
    ) -> PyResult<()> {
        let token_encoding = self
            .tokenizer
            .encode(top_token, false)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let token_ids = token_encoding.get_ids();
        let token_id = token_ids.first().copied().unwrap_or(0);

        let meta = FeatureMeta {
            top_token: top_token.to_string(),
            top_token_id: token_id,
            c_score,
            top_k: vec![larql_models::TopKEntry {
                token: top_token.to_string(),
                token_id,
                logit: c_score,
            }],
        };
        self.index.set_feature_meta(layer, feature, meta);
        Ok(())
    }

    /// Delete edges matching an entity (and optionally a relation).
    #[pyo3(signature = (entity, relation=None, layer=None))]
    fn delete(
        &mut self,
        entity: &str,
        relation: Option<&str>,
        layer: Option<usize>,
    ) -> PyResult<usize> {
        // Find matching features via describe
        let edges = self.describe(entity, "all", true)?;
        let mut deleted = 0;

        for edge in &edges {
            if let Some(r) = relation {
                if edge
                    .relation
                    .as_deref()
                    .map(|l| !l.eq_ignore_ascii_case(r))
                    .unwrap_or(true)
                {
                    continue;
                }
            }
            if let Some(l) = layer {
                if edge.layer != l {
                    continue;
                }
            }
            self.index.delete_feature_meta(edge.layer, edge.feature);
            deleted += 1;
        }

        Ok(deleted)
    }

    // ══════════════════════════════════════════════
    //  Config / Stats
    // ══════════════════════════════════════════════

    /// Return vindex stats as a dict.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("model", &self.config.model)?;
        dict.set_item("family", &self.config.family)?;
        dict.set_item("num_layers", self.config.num_layers)?;
        dict.set_item("hidden_size", self.config.hidden_size)?;
        dict.set_item("intermediate_size", self.config.intermediate_size)?;
        dict.set_item("vocab_size", self.config.vocab_size)?;
        dict.set_item("embed_scale", self.config.embed_scale)?;
        dict.set_item("dtype", self.config.dtype.to_string())?;
        dict.set_item("total_gate_vectors", self.index.total_gate_vectors())?;
        dict.set_item("total_down_meta", self.index.total_down_meta())?;
        dict.set_item("is_mmap", self.index.is_mmap())?;
        if let Some(ref rc) = self.classifier {
            dict.set_item("num_clusters", rc.num_clusters())?;
            dict.set_item("num_probe_labels", rc.num_probe_labels())?;
        }

        if let Some(ref bands) = self.config.layer_bands {
            let bands_dict = PyDict::new(py);
            bands_dict.set_item("syntax", (bands.syntax.0, bands.syntax.1))?;
            bands_dict.set_item("knowledge", (bands.knowledge.0, bands.knowledge.1))?;
            bands_dict.set_item("output", (bands.output.0, bands.output.1))?;
            dict.set_item("layer_bands", bands_dict)?;
        }

        Ok(dict)
    }

    /// Layer bands (syntax, knowledge, output) if available.
    fn layer_bands<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        match &self.config.layer_bands {
            None => Ok(None),
            Some(bands) => {
                let dict = PyDict::new(py);
                dict.set_item("syntax", (bands.syntax.0, bands.syntax.1))?;
                dict.set_item("knowledge", (bands.knowledge.0, bands.knowledge.1))?;
                dict.set_item("output", (bands.output.0, bands.output.1))?;
                Ok(Some(dict))
            }
        }
    }

    // ══════════════════════════════════════════════
    //  INFER — full forward pass with walk FFN
    // ══════════════════════════════════════════════

    /// Run inference: full forward pass with vindex walk FFN.
    ///
    /// Model weights are mmap'd on first call and reused — zero-copy, fast.
    /// Subsequent calls reuse the cached weights (OS page cache warms up).
    ///
    /// Routes through `larql_inference::infer_patched`, which is also the
    /// entry point for the LQL `SELECT ... INFER` executor — the two paths
    /// produce byte-identical top-k predictions on any vindex. See ADR 0001
    /// (`docs/adr/0001-python-lql-infer-parity.md`).
    ///
    /// Args:
    ///     prompt: input text
    ///     top_k_predictions: number of top predictions to return (default 5)
    ///
    /// Returns:
    ///     List of (token, probability) tuples
    #[pyo3(signature = (prompt, top_k_predictions=5))]
    fn infer(&self, prompt: &str, top_k_predictions: usize) -> PyResult<Vec<(String, f64)>> {
        self.with_walk_model(|infer_state| {
            let encoding = self
                .tokenizer
                .encode(prompt, true)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            let result = larql_inference::infer_patched(
                &infer_state.weights,
                &self.tokenizer,
                &self.index,
                self.knn_store.as_ref(),
                &token_ids,
                top_k_predictions,
            );
            Ok(result.predictions)
        })
    }

    /// Layers that have at least one entry in the L0 KnnStore.
    ///
    /// Empty if the vindex has no `knn_store.bin` or it loaded as empty.
    /// Used by measurement scripts that probe stored-key cosines against
    /// held-out residuals without running the override themselves.
    fn knn_layers(&self) -> Vec<usize> {
        self.knn_store
            .as_ref()
            .map(|s| s.layers())
            .unwrap_or_default()
    }

    /// Total number of entries across all layers in the L0 KnnStore.
    fn knn_len(&self) -> usize {
        self.knn_store.as_ref().map(|s| s.len()).unwrap_or(0)
    }

    /// Top-k cosine-similarity query against the L0 KnnStore at a single
    /// layer. Returns `(entity, relation, target_token, cosine)` tuples
    /// sorted descending by cosine.
    ///
    /// `residual` is the query vector — L2-normalisation is handled inside
    /// `query_knn`. Typical usage: capture residuals via `infer_trace`, then
    /// probe each layer in `knn_layers()` to measure the negative-mass
    /// distribution of held-out prompts against stored keys.
    #[pyo3(signature = (residual, layer, k=2))]
    fn knn_query(
        &self,
        residual: numpy::PyReadonlyArray1<f32>,
        layer: usize,
        k: usize,
    ) -> PyResult<Vec<(String, String, String, f32)>> {
        let store = match self.knn_store.as_ref() {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };
        let slice = residual.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("residual must be contiguous: {e}"))
        })?;
        let hits = store.query_knn(layer, slice, k);
        Ok(hits
            .into_iter()
            .map(|(entry, cos)| {
                (
                    entry.entity.clone(),
                    entry.relation.clone(),
                    entry.target_token.clone(),
                    cos,
                )
            })
            .collect())
    }

    /// Per-fact target-delta optimisation (MEMIT phase 3).
    ///
    /// Returns (delta_array, baseline_loss, final_loss). Currently only
    /// install_layer = n_layers-1 is supported; mid-layer backward
    /// through attention+FFN is pending.
    #[pyo3(signature = (prompt, target, install_layer, steps=60, lr=0.5, kl_weight=0.0625))]
    #[allow(clippy::too_many_arguments)]
    fn optimise_target_delta<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        target: &str,
        install_layer: usize,
        steps: usize,
        lr: f32,
        kl_weight: f32,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, f32, f32)> {
        self.with_walk_model(|infer_state| {
            let prompt_enc = self
                .tokenizer
                .encode(prompt, true)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let prompt_ids: Vec<u32> = prompt_enc.get_ids().to_vec();
            let target_spaced = format!(" {target}");
            let target_enc = self
                .tokenizer
                .encode(target_spaced.as_str(), false)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let target_id: u32 = target_enc.get_ids().first().copied().unwrap_or(0);

            let opts = larql_inference::TargetDeltaOpts {
                steps,
                lr,
                kl_weight,
                normalise: false,
            };
            let result = larql_inference::forward::target_delta::optimise_target_delta(
                &infer_state.weights,
                &prompt_ids,
                target_id,
                install_layer,
                opts,
            )
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

            let delta_vec = result.delta.to_vec();
            let delta_np = numpy::PyArray1::from_vec(py, delta_vec);
            Ok((delta_np, result.baseline_loss, result.final_loss))
        })
    }

    /// Run inference and capture per-layer residuals — the actual query
    /// vectors the walk FFN's `gate_knn` operates on at each layer
    /// (post-attention, post-RMSNorm, last-token position).
    ///
    /// Routes through `larql_inference::infer_patched` — same pipeline as
    /// `infer()` and the LQL `SELECT ... INFER` executor, so the returned
    /// predictions match those surfaces byte-for-byte (ADR 0001).
    ///
    /// Residuals are returned as `(layer, array)` tuples because the walk
    /// FFN only emits residuals for layers with vindex features — positional
    /// indexing does not correspond to layer number. Iterate:
    ///
    ///     for layer, r in residuals:
    ///         ...
    ///
    /// Returns:
    ///   (predictions, residuals) where
    ///     predictions: list of (token, probability) tuples
    ///     residuals:   list of (layer_index, (hidden_size,) numpy array)
    #[pyo3(signature = (prompt, top_k_predictions=5))]
    #[allow(clippy::type_complexity)]
    fn infer_trace<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        top_k_predictions: usize,
    ) -> PyResult<(Vec<(String, f64)>, Vec<(usize, Bound<'py, PyArray1<f32>>)>)> {
        self.with_walk_model(|infer_state| {
            let encoding = self
                .tokenizer
                .encode(prompt, true)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            let result = larql_inference::infer_patched(
                &infer_state.weights,
                &self.tokenizer,
                &self.index,
                self.knn_store.as_ref(),
                &token_ids,
                top_k_predictions,
            );

            let residuals: Vec<(usize, Bound<'py, PyArray1<f32>>)> = result
                .residuals
                .into_iter()
                .map(|(layer, vec)| (layer, ndarray::Array1::from_vec(vec).into_pyarray(py)))
                .collect();

            Ok((result.predictions, residuals))
        })
    }

    /// Find features whose down weight vectors project toward a target token.
    ///
    /// For each feature at the given layers, computes:
    ///   score = lm_head[token_id] · down_weight[layer, feature]
    ///
    /// Returns list of (layer, feature, score, top_token) sorted by score descending.
    /// Only returns features with score > 0.
    #[pyo3(signature = (target, layers=None, top_k=20))]
    fn find_features_by_target(
        &self,
        target: &str,
        layers: Option<Vec<usize>>,
        top_k: usize,
    ) -> PyResult<Vec<(usize, usize, f32, String)>> {
        self.with_walk_model(|infer_state| {
            let weights = &infer_state.weights;

            let encoding = self
                .tokenizer
                .encode(target, false)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let token_ids = encoding.get_ids();
            if token_ids.is_empty() {
                return Ok(vec![]);
            }
            let target_id = token_ids[0] as usize;
            let lm_head_row = weights.lm_head.row(target_id);

            let scan_layers = layers.unwrap_or_else(|| self.index.loaded_layers());
            let mut results: Vec<(usize, usize, f32, String)> = Vec::new();

            for &layer in &scan_layers {
                let arch = &*weights.arch;
                let down_key = arch.ffn_down_key(layer);
                let down_weights = match weights.tensors.get(&down_key) {
                    Some(w) => w,
                    None => continue,
                };
                let num_features = down_weights.shape()[0];

                for feat in 0..num_features {
                    let down_row = down_weights.row(feat);
                    let score: f32 = lm_head_row
                        .iter()
                        .zip(down_row.iter())
                        .map(|(a, b)| a * b)
                        .sum();

                    if score > 0.0 {
                        let token = self
                            .index
                            .feature_meta(layer, feat)
                            .map(|m| m.top_token.clone())
                            .unwrap_or_default();
                        results.push((layer, feat, score, token));
                    }
                }
            }

            results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(top_k);
            Ok(results)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Vindex(model='{}', layers={}, hidden={}, features={})",
            self.config.model,
            self.config.num_layers,
            self.config.hidden_size,
            self.index.total_gate_vectors()
        )
    }
}
