//! Cascade trie probe — PCA-32 + Logistic Regression classifier trained on
//! last-position hidden states at a fixed transformer layer (~15% depth).
//!
//! Loaded from a JSON file exported by `experiments/export_trie_probe.py`.
//! The exported file contains PCA components and LR weights; inference is
//! pure arithmetic — no Python dependency at runtime.
//!
//! ## Probe file lookup
//!
//! Probes are model-specific (each model has its own hidden geometry). Use
//! [`CascadeTrie::find`] to resolve a probe path for a given model id; it
//! consults `LARQL_PROBE_PATH` and `LARQL_PROBE_DIR` env overrides before
//! falling back to caller-supplied search directories.

use std::path::{Path, PathBuf};

use serde::Deserialize;

// ── Serialised probe format ───────────────────────────────────────────────────

#[derive(Deserialize)]
struct ProbeFile {
    layer: usize,
    hidden_size: usize,
    n_components: usize,
    routes: Vec<String>,
    pca_mean: Vec<f64>,
    pca_components: Vec<Vec<f64>>, // [n_components, hidden_size]
    lr_coef: Vec<Vec<f64>>,        // [n_classes, n_components]
    lr_intercept: Vec<f64>,        // [n_classes]
    lr_classes: Vec<String>,       // route name per LR class index
}

// ── Public API ────────────────────────────────────────────────────────────────

/// A loaded cascade trie probe.
///
/// Call `classify(hidden)` with the last-position hidden state (as `f32` slice,
/// length = hidden_size) from transformer layer `self.layer()`.
pub struct CascadeTrie {
    pub layer: usize,
    hidden_size: usize,
    routes: Vec<String>,
    /// PCA: subtract mean, then multiply by components.
    pca_mean: Vec<f32>,
    /// [n_components, hidden_size] row-major.
    pca_components: Vec<f32>,
    n_components: usize,
    /// LR: [n_classes, n_components] row-major.
    lr_coef: Vec<f32>,
    lr_intercept: Vec<f32>,
    lr_classes: Vec<String>,
}

impl CascadeTrie {
    /// Load from a JSON file exported by `export_trie_probe.py`.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        let p: ProbeFile = serde_json::from_str(&text)?;

        // Flatten 2D vecs to row-major 1D for BLAS-free dot products.
        let pca_components: Vec<f32> = p
            .pca_components
            .into_iter()
            .flatten()
            .map(|v| v as f32)
            .collect();
        let lr_coef: Vec<f32> = p.lr_coef.into_iter().flatten().map(|v| v as f32).collect();

        Ok(Self {
            layer: p.layer,
            hidden_size: p.hidden_size,
            routes: p.routes,
            pca_mean: p.pca_mean.into_iter().map(|v| v as f32).collect(),
            pca_components,
            n_components: p.n_components,
            lr_coef,
            lr_intercept: p.lr_intercept.into_iter().map(|v| v as f32).collect(),
            lr_classes: p.lr_classes,
        })
    }

    /// Classify a hidden state vector → route label (e.g. `"arithmetic"`).
    ///
    /// `hidden` must have length == `self.hidden_size`.
    /// Returns `"unknown"` if the slice length doesn't match.
    pub fn classify<'a>(&'a self, hidden: &[f32]) -> &'a str {
        if hidden.len() != self.hidden_size {
            return "unknown";
        }

        // ── PCA projection ──
        // z[k] = dot(hidden - mean, components[k])
        let mut z = vec![0.0f32; self.n_components];
        for (k, z_k) in z.iter_mut().enumerate() {
            let row = &self.pca_components[k * self.hidden_size..(k + 1) * self.hidden_size];
            let mut dot = 0.0f32;
            for i in 0..self.hidden_size {
                dot += (hidden[i] - self.pca_mean[i]) * row[i];
            }
            *z_k = dot;
        }

        // ── LR decision: argmax of (coef @ z + intercept) ──
        let n_classes = self.lr_classes.len();
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..n_classes {
            let row = &self.lr_coef[c * self.n_components..(c + 1) * self.n_components];
            let score: f32 =
                row.iter().zip(z.iter()).map(|(w, x)| w * x).sum::<f32>() + self.lr_intercept[c];
            if score > best_score {
                best_score = score;
                best_idx = c;
            }
        }

        &self.lr_classes[best_idx]
    }

    /// All route labels the probe was trained on.
    pub fn routes(&self) -> &[String] {
        &self.routes
    }

    /// Filesystem-safe slug for a model id (`google/gemma-3-4b-it` →
    /// `google--gemma-3-4b-it`). Matches `experiments/export_trie_probe.py`.
    pub fn slug(model_id: &str) -> String {
        model_id.replace('/', "--")
    }

    /// Standard probe filename for a model id.
    pub fn filename_for(model_id: &str) -> String {
        format!("cascade_trie_{}_probe.json", Self::slug(model_id))
    }

    /// Resolve a probe path for `model_id` by searching a precedence chain.
    ///
    /// Search order:
    ///   1. `LARQL_PROBE_PATH` env var — used verbatim if set and the file exists.
    ///   2. `LARQL_PROBE_DIR` env var, joined with [`Self::filename_for`].
    ///   3. Each entry in `extra_dirs`, joined with [`Self::filename_for`].
    ///
    /// Returns `None` if no probe is found anywhere in the chain.
    pub fn find<I, P>(model_id: &str, extra_dirs: I) -> Option<PathBuf>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        Self::find_with_env(
            model_id,
            std::env::var("LARQL_PROBE_PATH").ok().map(PathBuf::from),
            std::env::var("LARQL_PROBE_DIR").ok().map(PathBuf::from),
            extra_dirs,
        )
    }

    /// Pure version of [`Self::find`] — env-var values are passed in instead
    /// of read from the process environment. Exposed so tests can exercise
    /// the precedence chain without mutating shared env state (which would
    /// race with parallel tests).
    pub fn find_with_env<I, P>(
        model_id: &str,
        env_path: Option<PathBuf>,
        env_dir: Option<PathBuf>,
        extra_dirs: I,
    ) -> Option<PathBuf>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        if let Some(path) = env_path {
            if path.is_file() {
                return Some(path);
            }
        }
        let filename = Self::filename_for(model_id);
        if let Some(dir) = env_dir {
            let path = dir.join(&filename);
            if path.is_file() {
                return Some(path);
            }
        }
        for dir in extra_dirs {
            let path = dir.as_ref().join(&filename);
            if path.is_file() {
                return Some(path);
            }
        }
        None
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn slug_replaces_slashes() {
        assert_eq!(
            CascadeTrie::slug("google/gemma-3-4b-it"),
            "google--gemma-3-4b-it"
        );
        assert_eq!(CascadeTrie::slug("a/b/c"), "a--b--c");
        assert_eq!(CascadeTrie::slug("noslash"), "noslash");
    }

    #[test]
    fn filename_for_uses_slug() {
        assert_eq!(
            CascadeTrie::filename_for("google/gemma-3-4b-it"),
            "cascade_trie_google--gemma-3-4b-it_probe.json",
        );
    }

    #[test]
    fn find_returns_none_when_nothing_matches() {
        // Don't pollute env: verify by passing only a non-existent dir.
        let dir = std::env::temp_dir().join("larql-nonexistent-probe-dir-xyz");
        let r = CascadeTrie::find("does/not/exist", [&dir]);
        assert!(r.is_none());
    }

    #[test]
    fn find_resolves_from_extra_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mid = "test/model";
        let path = dir.path().join(CascadeTrie::filename_for(mid));
        std::fs::write(&path, "{}").expect("write");

        let found = CascadeTrie::find(mid, [dir.path()]).expect("found");
        assert_eq!(found, path);
    }

    #[test]
    fn find_extra_dirs_searched_in_order() {
        let first = tempfile::tempdir().expect("tempdir");
        let second = tempfile::tempdir().expect("tempdir");
        let mid = "test/order";
        let in_second = second.path().join(CascadeTrie::filename_for(mid));
        std::fs::write(&in_second, "{}").expect("write");

        // First dir is empty; lookup should fall through to second.
        let found = CascadeTrie::find(mid, [first.path(), second.path()]).expect("found");
        assert_eq!(found, in_second);
    }

    #[test]
    fn find_with_env_path_wins_over_dir_and_extra() {
        // env_path → returned regardless of env_dir / extra_dirs.
        let f = tempfile::NamedTempFile::new().expect("tempfile");
        let p = f.path().to_path_buf();

        let other = tempfile::tempdir().expect("tempdir");
        std::fs::write(other.path().join(CascadeTrie::filename_for("m")), "{}").unwrap();

        let found = CascadeTrie::find_with_env(
            "m",
            Some(p.clone()),
            Some(other.path().to_path_buf()),
            [other.path()],
        )
        .expect("found");
        assert_eq!(found, p);
    }

    #[test]
    fn find_with_env_path_falls_through_when_missing() {
        // env_path set to a non-existent file → should not be returned;
        // env_dir + filename should resolve next.
        let dir = tempfile::tempdir().expect("tempdir");
        let mid = "x/y";
        let resolved = dir.path().join(CascadeTrie::filename_for(mid));
        std::fs::write(&resolved, "{}").unwrap();

        let found = CascadeTrie::find_with_env(
            mid,
            Some(PathBuf::from("/this/file/should/not/exist.json")),
            Some(dir.path().to_path_buf()),
            std::iter::empty::<PathBuf>(),
        )
        .expect("found");
        assert_eq!(found, resolved);
    }

    #[test]
    fn find_with_env_dir_wins_over_extra() {
        // env_dir → preferred over extra_dirs.
        let env_dir = tempfile::tempdir().expect("tempdir");
        let extra_dir = tempfile::tempdir().expect("tempdir");
        let mid = "m/n";
        let in_env = env_dir.path().join(CascadeTrie::filename_for(mid));
        let in_extra = extra_dir.path().join(CascadeTrie::filename_for(mid));
        std::fs::write(&in_env, "{}").unwrap();
        std::fs::write(&in_extra, "{}").unwrap();

        let found = CascadeTrie::find_with_env(
            mid,
            None,
            Some(env_dir.path().to_path_buf()),
            [extra_dir.path()],
        )
        .expect("found");
        assert_eq!(found, in_env);
    }

    #[test]
    fn find_with_env_dir_missing_filename_falls_through_to_extra() {
        // env_dir set but the filename isn't there → fall through to extra_dirs.
        let env_dir = tempfile::tempdir().expect("tempdir"); // empty
        let extra_dir = tempfile::tempdir().expect("tempdir");
        let mid = "z/w";
        let in_extra = extra_dir.path().join(CascadeTrie::filename_for(mid));
        std::fs::write(&in_extra, "{}").unwrap();

        let found = CascadeTrie::find_with_env(
            mid,
            None,
            Some(env_dir.path().to_path_buf()),
            [extra_dir.path()],
        )
        .expect("found");
        assert_eq!(found, in_extra);
    }

    #[test]
    fn find_with_env_returns_none_when_nothing_resolves() {
        let none = CascadeTrie::find_with_env(
            "no/where",
            Some(PathBuf::from("/nope/path")),
            Some(PathBuf::from("/nope/dir")),
            std::iter::empty::<PathBuf>(),
        );
        assert!(none.is_none());
    }
}
