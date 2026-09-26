use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use super::statistic::validate_values;
use super::{
    json_digest, refused, valid_digest, CalibrationSite, Population, StatisticKind,
    CAPTURE_REVISION, SCHEMA,
};
use crate::error::VindexError;

/// Complete expected context, derived by PreparedCalibration, never inferred
/// from an artifact's own declarations when admitting that artifact for reuse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationKey {
    pub source_image_sha256: String,
    pub candidate_prefix_sha256: String,
    pub execution_sha256: String,
    pub site: CalibrationSite,
    pub bank_name: String,
    pub bank_sha256: String,
    pub tokenizer_sha256: String,
    /// Existing teacher-forced-token-bank/v1 authority when imported through
    /// CalibrationBank::from_token_bank; the mask envelope has its own digest.
    pub token_bank_id: Option<String>,
    pub population: Population,
    pub statistic: StatisticKind,
    pub samples: u64,
    pub capture_revision: String,
}

impl CalibrationKey {
    pub(crate) fn validate(&self) -> Result<(), VindexError> {
        if [
            &self.source_image_sha256,
            &self.candidate_prefix_sha256,
            &self.execution_sha256,
            &self.bank_sha256,
            &self.tokenizer_sha256,
        ]
        .iter()
        .any(|d| !valid_digest(d))
            || self.samples == 0
            || self.site.rows == 0
            || self.bank_name.trim().is_empty()
            || self.capture_revision != CAPTURE_REVISION
        {
            return Err(refused("invalid capture identity, samples or revision"));
        }
        self.statistic.elements(self.site.width)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationManifest {
    pub schema: String,
    pub key: CalibrationKey,
    /// Binary payload is row-major little-endian f64 raw sums, with no damping,
    /// centering, sample normalization or factor of two.
    pub numeric: String,
    pub payload_bytes: u64,
    pub payload_sha256: String,
    pub binding_sha256: String,
}

impl CalibrationManifest {
    pub(crate) fn binding(&self) -> Result<String, VindexError> {
        json_digest(&(
            &self.schema,
            &self.key,
            &self.numeric,
            self.payload_bytes,
            &self.payload_sha256,
        ))
    }
}

pub struct CalibrationArtifact {
    manifest: CalibrationManifest,
    values: Vec<f64>,
}

impl CalibrationArtifact {
    pub(super) fn new(key: CalibrationKey, values: Vec<f64>) -> Result<Self, VindexError> {
        key.validate()?;
        validate_values(key.statistic, key.site.width, &values)?;
        let mut hash = Sha256::new();
        for value in &values {
            hash.update(value.to_le_bytes());
        }
        let mut manifest = CalibrationManifest {
            schema: SCHEMA.into(),
            key,
            numeric: "f64-le-raw-sum".into(),
            payload_bytes: (values.len() as u64)
                .checked_mul(8)
                .ok_or_else(|| refused("payload length overflow"))?,
            payload_sha256: format!("{:x}", hash.finalize()),
            binding_sha256: String::new(),
        };
        manifest.binding_sha256 = manifest.binding()?;
        Ok(Self { manifest, values })
    }

    pub fn manifest(&self) -> &CalibrationManifest {
        &self.manifest
    }
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Move the single statistic into its consumer without cloning a dense H.
    pub(crate) fn into_parts(self) -> (CalibrationManifest, Vec<f64>) {
        (self.manifest, self.values)
    }

    /// Fresh directory only. Manifest is written last; failures cannot leave a
    /// complete-looking artifact or overwrite an earlier calibration run.
    pub fn write(&self, directory: &Path) -> Result<(), VindexError> {
        fs::create_dir(directory)?;
        let mut payload =
            std::io::BufWriter::new(File::create_new(directory.join("statistics.f64"))?);
        for value in &self.values {
            payload.write_all(&value.to_le_bytes())?;
        }
        payload.flush()?;
        payload.get_ref().sync_all()?;
        let mut manifest = File::create_new(directory.join("manifest.json"))?;
        manifest.write_all(&serde_json::to_vec_pretty(&self.manifest).map_err(refused)?)?;
        manifest.sync_all()?;
        Ok(())
    }

    /// The expected key must come from the intended capture context. Matching a
    /// self-declared digest alone proves integrity, not applicability or trust.
    pub fn read(directory: &Path, expected: &CalibrationKey) -> Result<Self, VindexError> {
        expected.validate()?;
        let manifest_file = File::open(directory.join("manifest.json"))?;
        if manifest_file.metadata()?.len() > 1024 * 1024 {
            return Err(refused("oversized calibration manifest"));
        }
        let manifest: CalibrationManifest =
            serde_json::from_reader(manifest_file).map_err(refused)?;
        if manifest.schema != SCHEMA || manifest.numeric != "f64-le-raw-sum" {
            return Err(refused(
                "unsupported calibration schema or numeric representation",
            ));
        }
        if &manifest.key != expected {
            return Err(refused(
                "calibration context mismatch (source, prefix, site, bank or recipe inputs)",
            ));
        }
        if manifest.binding_sha256 != manifest.binding()? {
            return Err(refused("manifest binding digest mismatch"));
        }
        let n = expected.statistic.elements(expected.site.width)?;
        let bytes = (n as u64)
            .checked_mul(8)
            .ok_or_else(|| refused("payload size overflow"))?;
        let file = File::open(directory.join("statistics.f64"))?;
        if manifest.payload_bytes != bytes || file.metadata()?.len() != bytes {
            return Err(refused("statistic payload length mismatch"));
        }
        let mut reader = std::io::BufReader::new(file);
        let mut values = Vec::new();
        values.try_reserve_exact(n).map_err(refused)?;
        for _ in 0..n {
            let mut word = [0; 8];
            reader.read_exact(&mut word)?;
            values.push(f64::from_le_bytes(word));
        }
        let checked = Self::new(expected.clone(), values)?;
        if checked.manifest != manifest {
            return Err(refused("statistic payload digest mismatch"));
        }
        Ok(checked)
    }
}
