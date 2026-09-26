//! What one execution actually RAN, as the executor's own record.
//!
//! V3-OBS-1's cross-backend witness on real Granite found two backends
//! producing an identical observation stream and materially different
//! values — and the difference was explained, not by the numbers, but
//! by the realizations each image had pinned: 121 projections
//! requantised to Q8 on one side, exact f32 on the other. Aligning the
//! realizations collapsed the disagreement to summation order. So a
//! carrier value, a projected coordinate or a selected logit is only
//! comparable across two runs when each run's record says what its
//! image pinned and which arithmetic arm its process resolved. The
//! arm is process-global (`arithmetic_arm()` is a `OnceLock`) and lives
//! in no operand record, which is why this record exists.
//!
//! Everything here is an OBSERVED fact of a prepared image or of the
//! running process — never the environment value that asked for it. A
//! cap that did not take effect is invisible here by design, exactly
//! as it is invisible in the pinned records.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::cpu::physical::{arithmetic_arm, kquant_execution, ArithmeticArm, KQuantExecution};
use super::observe_stats::{BasisIdentity, StatsObserver, NORM_METHOD, PROBE_METHOD};
use super::prepared::PreparedOperands;

/// One class of pinned realization: every operand that shares a stored
/// representation, a codec identity and a pinned physical form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RealizationClass {
    /// The stored representation label the operands resolved from.
    pub representation: String,
    /// `family@revision` of the codec that claimed the bytes; `None`
    /// for a label no codec claims.
    pub codec: Option<String>,
    /// The executor's own spelling of the pinned `RealizationId` —
    /// backend, form and physical plan — as the selection recorded it.
    pub form: String,
    pub operands: usize,
}

/// The executor-side half of a run's provenance: what the prepared
/// image pinned and what the process resolved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutionProvenance {
    /// The lowering provider(s) that pinned the image, as
    /// `family@revision`. One entry on every image today; a list so a
    /// mixed image could never be reported as a single provider.
    pub lowering: Vec<String>,
    /// Sorted by representation, codec and form, so two images that
    /// pinned the same things serialise identically.
    pub realizations: Vec<RealizationClass>,
    /// The activation arithmetic this PROCESS resolved — not a field of
    /// any operand record, and the reason a per-run record is needed.
    pub arithmetic_arm: ArithmeticArm,
    /// How a stored K-quant pack executes in this process.
    pub kquant_execution: KQuantExecution,
}

impl ExecutionProvenance {
    /// Read off a prepared image and the running process.
    pub fn of(ops: &PreparedOperands) -> Self {
        let mut lowering: Vec<String> = Vec::new();
        let mut classes: BTreeMap<(String, Option<String>, String), usize> = BTreeMap::new();
        for record in ops.realizations() {
            let provider = format!(
                "{}@{}",
                record.lowering_provider.family, record.lowering_provider.revision
            );
            if !lowering.contains(&provider) {
                lowering.push(provider);
            }
            let codec = record
                .codec_provider
                .as_ref()
                .map(|c| format!("{}@{}", c.family, c.revision));
            let form = format!("{:?}", record.selection.realization);
            *classes
                .entry((record.representation.clone(), codec, form))
                .or_default() += 1;
        }
        lowering.sort();
        Self {
            lowering,
            realizations: classes
                .into_iter()
                .map(
                    |((representation, codec, form), operands)| RealizationClass {
                        representation,
                        codec,
                        form,
                        operands,
                    },
                )
                .collect(),
            arithmetic_arm: arithmetic_arm(),
            kquant_execution: kquant_execution(),
        }
    }

    /// The number of operands the image pinned, summed over classes.
    pub fn operands(&self) -> usize {
        self.realizations.iter().map(|c| c.operands).sum()
    }

    /// SHA-256 over the canonical serialisation, as lowercase hex. Two
    /// runs whose fingerprints match executed the same RECORDED
    /// realizations under the same recorded arm. That is what this
    /// record can establish; it is not a proof that no unrecorded
    /// nondeterminism exists, so any residual difference between two
    /// matching runs is measured, never assumed away.
    pub fn fingerprint(&self) -> String {
        fingerprint_of(self)
    }
}

/// A run's complete provenance for observation: the execution half
/// plus the observer's own identities — which basis projected the
/// coordinates, which tokens the probe read, and by which methods.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RunProvenance {
    pub execution: ExecutionProvenance,
    pub basis: Option<BasisIdentity>,
    pub probe_tokens: Option<Vec<u32>>,
    pub norm_method: &'static str,
    pub probe_method: &'static str,
}

impl RunProvenance {
    /// Compose from the image and, where one is running, the stats
    /// observer whose rows the record will accompany.
    pub fn new(execution: ExecutionProvenance, observer: Option<&StatsObserver>) -> Self {
        Self {
            execution,
            basis: observer.map(|o| o.basis().identity()),
            probe_tokens: observer.and_then(|o| o.probe().map(|p| p.tokens().to_vec())),
            norm_method: NORM_METHOD,
            probe_method: PROBE_METHOD,
        }
    }

    /// SHA-256 over the canonical serialisation of the whole record.
    pub fn fingerprint(&self) -> String {
        fingerprint_of(self)
    }

    /// The record as JSON, for a runner to embed in its envelope or
    /// receipt verbatim.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("a provenance record serialises")
    }
}

fn fingerprint_of<T: Serialize>(value: &T) -> String {
    let json = serde_json::to_string(value).expect("a provenance record serialises");
    let digest = Sha256::digest(json.as_bytes());
    digest
        .iter()
        .fold(String::with_capacity(digest.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}
