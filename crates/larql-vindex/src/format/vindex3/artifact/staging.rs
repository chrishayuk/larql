//! What staging a repo's headers cost, and what it stands in for.

use crate::error::VindexError;

/// The figures one staging pass produced.
///
/// Headers and metadata are separate fields rather than one total because
/// they scale differently and quoting the header figure alone understates
/// the transfer — a tokenizer can outweigh every shard header put
/// together. GLM-5.3-Flash stages 10.68 MB of headers beside 28.71 MB of
/// metadata, so the header number alone is off by a factor of four.
pub struct StagingReport {
    pub header_bytes: u64,
    pub metadata_bytes: u64,
    pub shards: usize,
    /// Payload the staged HEADERS declare, or the census error.
    ///
    /// Never the shard index's `metadata.total_size`. The two disagree
    /// whenever the source model tied weights: HF computes `total_size`
    /// from deduplicated parameter storage, so it declares a tied
    /// embedding once while the file serialises it twice. granite-4.2-3b
    /// declares 6,805,672,960 bytes against 7,319,475,200 of headers —
    /// short by exactly one 513,802,240-byte member.
    ///
    /// A census failure is carried rather than raised: the encode reads
    /// the same headers and fails with a better message.
    pub payload_bytes: Result<u64, VindexError>,
    /// What the shard index declares, when it declares one.
    ///
    /// Kept so a caller can state the difference against
    /// [`Self::payload_bytes`] rather than let a silent 7% gap between
    /// "standing in for" and "fetched" read like a units bug.
    pub declared_total: Option<u64>,
}

impl StagingReport {
    /// Everything staged, headers and metadata together.
    pub fn staged_bytes(&self) -> u64 {
        self.header_bytes + self.metadata_bytes
    }

    /// By how much the shard index and the headers disagree, when they do.
    ///
    /// `None` when they agree, when the index declares nothing, or when
    /// the census failed — three different reasons there is nothing to
    /// report, and none of them is a difference of zero.
    pub fn index_disagreement(&self) -> Option<u64> {
        let payload = *self.payload_bytes.as_ref().ok()?;
        self.declared_total
            .filter(|declared| *declared != payload)
            .map(|declared| declared.abs_diff(payload))
    }
}

/// What a staging pass cost and what it stands in for, as JSON.
///
/// The one place this object's shape is written. Two front doors print
/// it — `vindex plan --json` and `POST /v1/plan` — and the server's
/// response body is defined as the CLI's document plus one serving
/// field, so two hand-written copies would make that parity a
/// coincidence rather than a property.
///
/// A pure function of the report rather than a method on
/// `ResolvedArtifact`, because an artifact with a remote origin can
/// only be built by staging a real repo: through that door the shape
/// could only ever be checked over the network, which is to say never.
/// `ResolvedArtifact::staging_json` is the two-line adapter.
pub(super) fn staging_report_json(
    name: &str,
    commit: Option<&str>,
    report: &StagingReport,
) -> serde_json::Value {
    serde_json::json!({
        "artifact": name,
        "commit": commit,
        "shards": report.shards,
        "staged": super::size(report.staged_bytes()),
        "headers": super::size(report.header_bytes),
        "metadata": super::size(report.metadata_bytes),
        "stands_in_for": report.payload_bytes.as_ref().ok().map(|b| super::size(*b)),
        // Stated only when the index disagrees with its own headers, so
        // the difference reads as a fact about the checkpoint rather than
        // a units bug in the report.
        "index_declares": report
            .declared_total
            .filter(|d| report.payload_bytes.as_ref().is_ok_and(|p| d != p))
            .map(super::size),
    })
}
