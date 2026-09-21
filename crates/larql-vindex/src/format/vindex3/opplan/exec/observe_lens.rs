//! V3-LENS-1: the true logit lens, as a consumer of the carrier tap
//! (`docs/v3-lens-1-logit-lens.md`).
//!
//! At every site it was armed for, the lens copies the carrier the next
//! layer would read — the write's `after`, times the layer scale where
//! the program applies one — and hands it to
//! [`PreparedOperands::head_logits`]: the prepared final norm and the
//! prepared output head, multiplier and softcap included, on the head
//! the image actually pinned. That is the same function the decode exit
//! calls, so the lens at the last layer IS the executor's logits rather
//! than something that agrees with them.
//!
//! Each armed site costs one full head pass per token. That is the
//! price of a true lens, and the lens counts it rather than hiding it:
//! a consumer that wants a cheap per-write statistic has the stats
//! observer; this one answers "what would the model say here".

use super::backend::PlanBackend;
use super::observe::{CarrierWriteRecord, StepEvent, StepObserver, SublayerSite};
use super::prepared::PreparedOperands;
use crate::error::VindexError;

/// The method every readout in this module was computed by: the
/// image's own final norm and output head, then a full log-softmax in
/// f64. A tuned lens or any other reader would be a different name.
pub const LENS_METHOD: &str = "head-v1";

/// Which layers a lens is armed on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LensLayers {
    All,
    /// Layers whose index is a multiple of the period.
    Every(usize),
    List(Vec<usize>),
}

/// Which sites a lens reads: a layer selection and which of the two
/// sublayer sites. The price is one head pass per armed site per token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LensSites {
    pub layers: LensLayers,
    pub attention: bool,
    pub ffn: bool,
}

impl LensSites {
    /// Every layer's FFN site: the layer outputs, which is what "the
    /// model at depth L" means unless a reader asks for more.
    pub fn every_ffn() -> Self {
        Self {
            layers: LensLayers::All,
            attention: false,
            ffn: true,
        }
    }

    /// `all`, `every:<k>`, or a comma-separated list of layer indices.
    pub fn parse_layers(spec: &str) -> Result<LensLayers, String> {
        let spec = spec.trim();
        if spec == "all" {
            return Ok(LensLayers::All);
        }
        if let Some(period) = spec.strip_prefix("every:") {
            let period: usize = period
                .trim()
                .parse()
                .map_err(|e| format!("`{spec}`: the period is not a number: {e}"))?;
            if period == 0 {
                return Err(format!("`{spec}`: the period must be at least 1"));
            }
            return Ok(LensLayers::Every(period));
        }
        let layers = spec
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|e| format!("`{spec}`: `{s}` is not a layer index: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if layers.is_empty() {
            return Err(format!("`{spec}`: no layers"));
        }
        Ok(LensLayers::List(layers))
    }

    pub fn armed(&self, layer: usize, site: SublayerSite) -> bool {
        let site_armed = match site {
            SublayerSite::Attention => self.attention,
            SublayerSite::Ffn => self.ffn,
        };
        site_armed
            && match &self.layers {
                LensLayers::All => true,
                LensLayers::Every(period) => layer.is_multiple_of(*period),
                LensLayers::List(list) => list.contains(&layer),
            }
    }
}

/// One declared token's standing at one site.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenReadout {
    pub id: u32,
    /// Under the full log-softmax over every logit, in f64.
    pub logprob: f64,
    /// One plus the number of logits strictly greater than this token's.
    pub rank: usize,
}

/// What the head said at one armed site.
#[derive(Debug, Clone, PartialEq)]
pub struct Readout {
    pub layer: usize,
    pub site: SublayerSite,
    pub position: usize,
    pub tokens: Vec<TokenReadout>,
    /// The top `k` ids by logit with their log-probabilities, when asked.
    pub top: Vec<(u32, f64)>,
    /// The full logits, kept only when the lens was built to retain them
    /// (witnesses); never persisted by the record.
    pub logits: Option<Vec<f32>>,
}

/// What [`readout_of`] returns: the declared tokens' standings and the
/// top ids with their log-probabilities.
pub type ReadoutParts = (Vec<TokenReadout>, Vec<(u32, f64)>);

/// The distribution's facts for `tokens` and the top `k`, from one set
/// of logits. Shared by every consumer that turns logits into a readout
/// so that "rank" and "logprob" mean one thing.
pub fn readout_of(
    logits: &[f32],
    tokens: &[u32],
    top_k: usize,
) -> Result<ReadoutParts, VindexError> {
    if logits.is_empty() {
        return Err(VindexError::Parse(
            "the head produced no logits".to_string(),
        ));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = logits
        .iter()
        .map(|&v| (f64::from(v) - f64::from(max)).exp())
        .sum::<f64>()
        .ln()
        + f64::from(max);
    let mut readouts = Vec::with_capacity(tokens.len());
    for &id in tokens {
        let index = usize::try_from(id).expect("a token id fits");
        let Some(&logit) = logits.get(index) else {
            return Err(VindexError::Parse(format!(
                "lens token {id} is outside the head's vocabulary of {}",
                logits.len()
            )));
        };
        let rank = 1 + logits.iter().filter(|&&v| v > logit).count();
        readouts.push(TokenReadout {
            id,
            logprob: f64::from(logit) - log_sum,
            rank,
        });
    }
    let mut top: Vec<(u32, f64)> = Vec::new();
    if top_k > 0 {
        let mut indexed: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (u32::try_from(i).expect("fits"), v))
            .collect();
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        top = indexed
            .into_iter()
            .take(top_k)
            .map(|(id, v)| (id, f64::from(v) - log_sum))
            .collect();
    }
    Ok((readouts, top))
}

/// What a recorder needs from a lens without knowing its backend type:
/// read one write, say how many head passes it cost, and say whether it
/// failed. `LogitLens` implements it; a record can hold any reader.
pub trait LensReader {
    fn read(&mut self, record: CarrierWriteRecord<'_>) -> Option<Readout>;
    fn head_passes(&self) -> usize;
    fn failure(&self) -> Option<&VindexError>;
    /// The tokens the reader was armed with, in readout order.
    fn tokens(&self) -> &[u32];
}

/// The lens: a consumer of the carrier tap over one prepared image.
pub struct LogitLens<'a, B: PlanBackend + ?Sized> {
    ops: &'a PreparedOperands,
    backend: &'a B,
    sites: LensSites,
    tokens: Vec<u32>,
    top_k: usize,
    retain_logits: bool,
    pub readouts: Vec<Readout>,
    /// Head passes performed: the lens's price, counted.
    pub head_passes: usize,
    /// The first failure, if the head refused or a token was outside the
    /// vocabulary; the lens stops reading once one is recorded, because
    /// an observer callback cannot return it.
    pub failure: Option<VindexError>,
}

impl<'a, B: PlanBackend + ?Sized> LogitLens<'a, B> {
    pub fn new(
        ops: &'a PreparedOperands,
        backend: &'a B,
        sites: LensSites,
        tokens: Vec<u32>,
        top_k: usize,
    ) -> Self {
        Self {
            ops,
            backend,
            sites,
            tokens,
            top_k,
            retain_logits: false,
            readouts: Vec::new(),
            head_passes: 0,
            failure: None,
        }
    }

    /// Keep every readout's full logits, for a witness that compares
    /// them to the executor's own.
    pub fn retaining_logits(mut self) -> Self {
        self.retain_logits = true;
        self
    }

    pub fn sites(&self) -> &LensSites {
        &self.sites
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Read one carrier through the head: the layer output, which is the
    /// write's `after` times the layer scale where the program has one.
    pub fn read(&mut self, record: CarrierWriteRecord<'_>) -> Option<Readout> {
        if self.failure.is_some() || !self.sites.armed(record.layer, record.site) {
            return None;
        }
        let scaled;
        let carrier: &[f32] = match record.layer_scale {
            Some(scale) => {
                scaled = record.after.iter().map(|v| v * scale).collect::<Vec<f32>>();
                &scaled
            }
            None => record.after,
        };
        self.head_passes += 1;
        let logits = match self.ops.head_logits(self.backend, carrier) {
            Ok(Some(logits)) => logits,
            Ok(None) => {
                self.failure = Some(VindexError::Parse(
                    "the image carries no output head, so there is nothing to read".to_string(),
                ));
                return None;
            }
            Err(e) => {
                self.failure = Some(e);
                return None;
            }
        };
        let (tokens, top) = match readout_of(&logits, &self.tokens, self.top_k) {
            Ok(r) => r,
            Err(e) => {
                self.failure = Some(e);
                return None;
            }
        };
        Some(Readout {
            layer: record.layer,
            site: record.site,
            position: record.position,
            tokens,
            top,
            logits: self.retain_logits.then_some(logits),
        })
    }
}

impl<B: PlanBackend + ?Sized> LensReader for LogitLens<'_, B> {
    fn read(&mut self, record: CarrierWriteRecord<'_>) -> Option<Readout> {
        LogitLens::read(self, record)
    }

    fn head_passes(&self) -> usize {
        self.head_passes
    }

    fn failure(&self) -> Option<&VindexError> {
        self.failure.as_ref()
    }

    fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

impl<B: PlanBackend + ?Sized> StepObserver for LogitLens<'_, B> {
    /// Structural events carry nothing the lens needs: the write record
    /// names its own layer, site and position.
    fn event(&mut self, _event: StepEvent) {}

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        if let Some(readout) = self.read(record) {
            self.readouts.push(readout);
        }
    }
}
