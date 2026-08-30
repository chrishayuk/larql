//! The Mamba2 mixer's declared geometry — the SSM sibling of
//! [`KdaGeometry`](super::KdaGeometry) and
//! [`LinearAttentionTopology`](crate::inventory::LinearAttentionTopology).
//!
//! Mamba2 (state-spaces SSD) is a third recurrence family, not a mode of
//! either existing one: a single fused `in_proj` emitting z|x|B|C|dt, one
//! depthwise causal conv over the x|B|C channels only, per-**head** scalar
//! decay (`A_log`), skip (`D`) and timestep bias (`dt_bias`), and a gated
//! RMSNorm between the state read-out and the output projection. Reading
//! it as Gated DeltaNet or KDA would bind the wrong tensors to the wrong
//! roles with plausible shapes.
//!
//! Every field is read from the checkpoint's own declaration; a partial
//! declaration is refused rather than completed with defaults — the same
//! contract the other two families hold.

use serde::{Deserialize, Serialize};

/// One side of the forward-time `dt` clamp (`time_step_limit`).
///
/// Transformers writes the unbounded side as a bare Python `Infinity`,
/// which the judged non-finite boundary
/// ([`super::nonfinite_json`]) parses as the string it spells. This type
/// is where that string becomes a semantic: an unbounded clamp side, as a
/// declared fact — never an IEEE infinity smuggled through JSON, which
/// strict serialization cannot round-trip.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DtBound {
    /// A finite clamp bound.
    Finite(f64),
    /// No clamp on this side (`Infinity` / `-Infinity` in the config).
    Unbounded,
}

impl DtBound {
    /// The judged reading of one `time_step_limit` element: a JSON number
    /// is a finite bound; the strings the non-finite boundary produces
    /// (`"Infinity"`, `"-Infinity"`) are declared unboundedness. Anything
    /// else — including `"NaN"`, which bounds nothing — is refused.
    pub fn from_declared(value: &serde_json::Value) -> Option<Self> {
        if let Some(v) = value.as_f64() {
            return Some(Self::Finite(v));
        }
        match value.as_str() {
            Some("Infinity") | Some("-Infinity") => Some(Self::Unbounded),
            _ => None,
        }
    }
}

/// Which key dialect declared the Mamba2 geometry.
///
/// Two lineages spell the same operator: transformers' HF conversion
/// (`num_heads`/`head_dim`/`conv_kernel`, e.g. `AntonV/mamba2-780m-hf`)
/// and the `mamba_ssm`-package lineage (`mamba2_num_heads`/
/// `mamba2_head_dim`/`mamba2_conv_kernel`, e.g. OuteAI's Mamba2Attn
/// hybrids). Recording which one was read — rather than silently
/// aliasing — is what lets a wrong dialect table be *found*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mamba2Dialect {
    /// transformers' key set.
    Hf,
    /// The `mamba_ssm`-package key set (`mamba2_*`-prefixed).
    MambaSsm,
}

/// One geometry field the declaration does not carry, filled from the
/// source family's own default — **recorded, never silent**. The value is
/// still subject to the same cross-field closure as a declared one, so a
/// wrong default is caught by the shapes it fails to close over.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mamba2FamilyDefault {
    /// The canonical field name (`n_groups`, `rms_norm`).
    pub key: String,
    /// The default taken, rendered as the config would spell it.
    pub value: String,
    /// Where the default is defined (`mamba_ssm Mamba2.__init__`).
    pub source: String,
}

/// How a [`Mamba2Geometry`] was read: the dialect, and every field that
/// came from a family default rather than the checkpoint's declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mamba2Provenance {
    pub dialect: Mamba2Dialect,
    /// Empty when every field was declared.
    pub family_defaults: Vec<Mamba2FamilyDefault>,
}

/// The Mamba2 mixer's declared geometry and forward-pass switches.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Mamba2Geometry {
    /// `state_size` — N, the SSM state width per group (128).
    pub state_size: usize,
    /// `num_heads` — heads the scalar decay/skip/timestep are declared
    /// per (48). The axis `A_log`, `D` and `dt_bias` are shaped along.
    pub num_heads: usize,
    /// `head_dim` — P, one head's slice of the inner width (64).
    pub head_dim: usize,
    /// `expand` — E, inner width as a multiple of `hidden_size` (2).
    pub expand: usize,
    /// `conv_kernel` — depthwise causal conv width over x|B|C, and the
    /// per-layer conv state length is `conv_kernel - 1` (4).
    pub conv_kernel: usize,
    /// `n_groups` — B/C projection group count (1).
    pub n_groups: usize,
    /// `chunk_size` — the SSD scan's chunk length (256). Algebraically
    /// result-invariant but not fp-invariant: chunking decides the
    /// accumulation order, so it is an execution fact, not a tuning knob.
    pub chunk_size: usize,
    /// `time_step_limit[0]` — the clamp under `dt` after softplus.
    pub dt_limit_min: DtBound,
    /// `time_step_limit[1]` — the clamp over `dt`; `Unbounded` on the
    /// released checkpoints (bare `Infinity` in the config).
    pub dt_limit_max: DtBound,
    /// `rms_norm` — whether the mixer carries the gated RMSNorm between
    /// state read-out and `out_proj` (evidenced by `mixer.norm.weight`).
    pub rms_norm: bool,
    /// `use_bias` — whether `in_proj`/`out_proj` carry biases.
    pub use_bias: bool,
    /// `use_conv_bias` — whether the depthwise conv carries a bias.
    pub use_conv_bias: bool,
}

impl Mamba2Geometry {
    /// Read the geometry from a (text) config object. All fields or none:
    /// a partial declaration is refused rather than completed, the same
    /// contract [`KdaGeometry::read`](super::KdaGeometry::read) holds.
    pub fn read(config: &serde_json::Value) -> Option<Self> {
        let dim = |key: &str| config[key].as_u64().map(|v| v as usize).filter(|v| *v > 0);
        let flag = |key: &str| config[key].as_bool();
        let limit = config["time_step_limit"].as_array()?;
        let [lo, hi] = limit.as_slice() else {
            return None;
        };
        Some(Self {
            state_size: dim("state_size")?,
            num_heads: dim("num_heads")?,
            head_dim: dim("head_dim")?,
            expand: dim("expand")?,
            conv_kernel: dim("conv_kernel")?,
            n_groups: dim("n_groups")?,
            chunk_size: dim("chunk_size")?,
            dt_limit_min: DtBound::from_declared(lo)?,
            dt_limit_max: DtBound::from_declared(hi)?,
            rms_norm: flag("rms_norm")?,
            use_bias: flag("use_bias")?,
            use_conv_bias: flag("use_conv_bias")?,
        })
    }

    /// Read the geometry in whichever dialect this checkpoint declares,
    /// with provenance. The HF spelling is tried first and admits no
    /// defaults; the `mamba_ssm` spelling fills exactly two fields the
    /// package never writes into configs (`n_groups`, `rms_norm`) from
    /// its own `__init__` defaults — each one recorded. A declared value
    /// always outranks the default.
    pub fn read_with_provenance(config: &serde_json::Value) -> Option<(Self, Mamba2Provenance)> {
        if let Some(geometry) = Self::read(config) {
            return Some((
                geometry,
                Mamba2Provenance {
                    dialect: Mamba2Dialect::Hf,
                    family_defaults: Vec::new(),
                },
            ));
        }
        Self::read_mamba_ssm(config)
    }

    /// The `mamba_ssm`-package dialect (OuteAI Mamba2Attn): three renamed
    /// geometry keys, `use_mamba2_bias` for the projection-bias switch,
    /// and two fields the dialect leaves to package defaults. Still
    /// all-or-nothing over the keys the dialect *does* spell.
    fn read_mamba_ssm(config: &serde_json::Value) -> Option<(Self, Mamba2Provenance)> {
        let dim = |key: &str| config[key].as_u64().map(|v| v as usize).filter(|v| *v > 0);
        let flag = |key: &str| config[key].as_bool();
        // The dialect is identified by its own spelling: without the
        // renamed head keys there is nothing to read as this dialect.
        let num_heads = dim("mamba2_num_heads")?;
        let head_dim = dim("mamba2_head_dim")?;
        let conv_kernel = dim("mamba2_conv_kernel")?;
        let limit = config["time_step_limit"].as_array()?;
        let [lo, hi] = limit.as_slice() else {
            return None;
        };
        let mut family_defaults = Vec::new();
        let mut defaulted = |key: &str, value: &str| {
            family_defaults.push(Mamba2FamilyDefault {
                key: key.to_string(),
                value: value.to_string(),
                source: "mamba_ssm Mamba2.__init__".to_string(),
            });
        };
        let n_groups = match dim("n_groups") {
            Some(declared) => declared,
            None => {
                defaulted("n_groups", "1");
                1
            }
        };
        let rms_norm = match flag("rms_norm") {
            Some(declared) => declared,
            None => {
                defaulted("rms_norm", "true");
                true
            }
        };
        Some((
            Self {
                state_size: dim("state_size")?,
                num_heads,
                head_dim,
                expand: dim("expand")?,
                conv_kernel,
                n_groups,
                chunk_size: dim("chunk_size")?,
                dt_limit_min: DtBound::from_declared(lo)?,
                dt_limit_max: DtBound::from_declared(hi)?,
                rms_norm,
                use_bias: flag("use_mamba2_bias")?,
                use_conv_bias: flag("use_conv_bias")?,
            },
            Mamba2Provenance {
                dialect: Mamba2Dialect::MambaSsm,
                family_defaults,
            },
        ))
    }

    /// D_inner — the mixer's inner width, `expand · hidden_size`.
    pub fn d_inner(self, hidden_size: usize) -> usize {
        self.expand * hidden_size
    }

    /// Channels the depthwise conv runs over: x|B|C =
    /// `d_inner + 2 · n_groups · state_size` — the conv weight's row
    /// count, and the width of the per-layer conv state.
    pub fn conv_dim(self, hidden_size: usize) -> usize {
        self.d_inner(hidden_size) + 2 * self.n_groups * self.state_size
    }

    /// Rows the fused `in_proj` emits: z|x|B|C|dt =
    /// `2 · d_inner + 2 · n_groups · state_size + num_heads`.
    ///
    /// Derived, never stored, so it cannot drift from the fields it is
    /// computed from. On mamba2-780m this closes exactly: `2·3072 +
    /// 2·1·128 + 48 = 6448`, the observed `mixer.in_proj.weight` row
    /// count.
    pub fn in_proj_rows(self, hidden_size: usize) -> usize {
        2 * self.d_inner(hidden_size) + 2 * self.n_groups * self.state_size + self.num_heads
    }

    /// Elements in one layer's SSM state: one `head_dim × state_size`
    /// matrix per head.
    pub fn state_elements(self) -> usize {
        self.num_heads * self.head_dim * self.state_size
    }

    /// Cross-field defects between the declared geometry and the
    /// component width it must close over. Empty when the declaration is
    /// internally consistent.
    pub fn geometry_defects(self, hidden_size: usize) -> Vec<String> {
        let mut defects = Vec::new();
        let d_inner = self.d_inner(hidden_size);
        if self.num_heads * self.head_dim != d_inner {
            defects.push(format!(
                "num_heads ({}) × head_dim ({}) = {} does not close over \
                 d_inner = expand ({}) × hidden_size ({}) = {}",
                self.num_heads,
                self.head_dim,
                self.num_heads * self.head_dim,
                self.expand,
                hidden_size,
                d_inner,
            ));
        }
        if !d_inner.is_multiple_of(self.n_groups) {
            defects.push(format!(
                "d_inner ({d_inner}) is not divisible by n_groups ({})",
                self.n_groups
            ));
        }
        defects
    }
}
