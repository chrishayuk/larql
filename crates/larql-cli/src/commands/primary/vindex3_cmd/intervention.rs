//! `vindex3 observe --intervene` and `--capture`: the two files the verb
//! reads and writes for V3-INTERVENE-1, and the observer that lets one
//! run feed both the recorder and a carrier capture.
//!
//! A DECLARATION names interventions. Each entry gives an address
//! (`layer`, `site`, `positions`), a `kind`, and for `add`/`replace` a
//! `vector`: an inline array (a literal), a reference into a capture file
//! (`{file, layer, site, position}` — the carrier another run wrote
//! there, with that run's identity as provenance), or the difference of
//! two such references (`{minuend, subtrahend}` — a literal whose hash is
//! computed here; the HEAD-1 CARRIED arm is `add` of `h14^B − h14^A`).
//! An optional `sha256` on an entry is CHECKED against the bytes; absent,
//! it is computed and still travels on the receipt.
//!
//! A CAPTURE FILE is what `--capture-out` writes: the run id and, per
//! captured address, the vector and its hash. A declaration that reads
//! one verifies the stored hash before trusting the bytes.

use std::path::{Path, PathBuf};

use larql_inference::vindex3::RunRecorder;
use larql_vindex::format::vindex3::opplan::exec::intervene::{
    vector_sha256, Address, CarrierCapture, Intervention, InterventionKind, InterventionPlan,
    VectorProvenance,
};
use larql_vindex::format::vindex3::opplan::exec::observe::{
    CarrierWriteRecord, StepEvent, StepObserver, SublayerSite,
};
use serde::{Deserialize, Serialize};

type BoxErr = Box<dyn std::error::Error>;

/// The declaration file.
#[derive(Debug, Deserialize)]
pub struct Declaration {
    pub interventions: Vec<Declared>,
}

/// One declared intervention, as the file spells it.
#[derive(Debug, Deserialize)]
pub struct Declared {
    pub layer: usize,
    pub site: String,
    pub positions: Vec<usize>,
    pub kind: String,
    #[serde(default)]
    pub vector: Option<VectorSource>,
    /// Checked against the vector's bytes when present.
    #[serde(default)]
    pub sha256: Option<String>,
}

/// Where a declared vector comes from.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum VectorSource {
    /// The vector itself.
    Literal(Vec<f32>),
    /// A carrier another run captured, by file and address.
    Captured(CapturedRef),
    /// `minuend − subtrahend`, both captured carriers.
    Difference {
        minuend: CapturedRef,
        subtrahend: CapturedRef,
    },
}

/// A reference into a capture file.
#[derive(Debug, Clone, Deserialize)]
pub struct CapturedRef {
    pub file: PathBuf,
    pub layer: usize,
    pub site: String,
    pub position: usize,
}

/// What `--capture-out` writes.
#[derive(Debug, Serialize, Deserialize)]
pub struct CaptureFile {
    pub run_id: String,
    pub captures: Vec<CaptureEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CaptureEntry {
    pub layer: usize,
    pub site: String,
    pub position: usize,
    pub sha256: String,
    pub values: Vec<f32>,
}

pub fn site(name: &str) -> Result<SublayerSite, BoxErr> {
    match name {
        "attention" => Ok(SublayerSite::Attention),
        "ffn" => Ok(SublayerSite::Ffn),
        other => Err(format!("unknown site `{other}`: expected `attention` or `ffn`").into()),
    }
}

pub fn site_name(site: SublayerSite) -> &'static str {
    match site {
        SublayerSite::Attention => "attention",
        SublayerSite::Ffn => "ffn",
    }
}

fn kind(name: &str) -> Result<InterventionKind, BoxErr> {
    match name {
        "zero" => Ok(InterventionKind::Zero),
        "add" => Ok(InterventionKind::Add),
        "replace" => Ok(InterventionKind::Replace),
        other => Err(format!(
            "unknown intervention kind `{other}`: expected `zero`, `add` or `replace`"
        )
        .into()),
    }
}

/// `layer:site:position[,layer:site:position…]`.
pub fn parse_addresses(list: &str) -> Result<Vec<(usize, SublayerSite, usize)>, BoxErr> {
    list.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|item| {
            let parts: Vec<&str> = item.split(':').collect();
            if parts.len() != 3 {
                return Err(
                    format!("capture address `{item}` is not `layer:site:position`").into(),
                );
            }
            Ok((parts[0].parse()?, site(parts[1])?, parts[2].parse()?))
        })
        .collect()
}

/// Read a declaration file into a plan. Relative capture-file paths
/// resolve against the declaration's own directory.
pub fn load_plan(path: &Path) -> Result<InterventionPlan, BoxErr> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read the declaration {}: {e}", path.display()))?;
    let declaration: Declaration = serde_json::from_str(&text)
        .map_err(|e| format!("the declaration {} does not parse: {e}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut plan = InterventionPlan::none();
    for entry in &declaration.interventions {
        plan = plan.with(resolve(entry, base)?)?;
    }
    Ok(plan)
}

fn resolve(entry: &Declared, base: &Path) -> Result<Intervention, BoxErr> {
    let address = Address::new(
        entry.layer,
        site(&entry.site)?,
        entry.positions.iter().copied(),
    )?;
    let kind = kind(&entry.kind)?;
    if kind == InterventionKind::Zero {
        if entry.vector.is_some() {
            return Err("a `zero` intervention carries no vector".into());
        }
        return Ok(Intervention::zero(address));
    }
    let (vector, provenance) = match entry.vector.as_ref() {
        None => return Err(format!("a `{}` intervention needs a `vector`", entry.kind).into()),
        Some(VectorSource::Literal(values)) => {
            let sha256 = vector_sha256(values);
            (values.clone(), VectorProvenance::Literal { sha256 })
        }
        Some(VectorSource::Captured(reference)) => {
            let (run_id, values) = read_capture(reference, base)?;
            let sha256 = vector_sha256(&values);
            (
                values,
                VectorProvenance::Captured {
                    run_id,
                    layer: reference.layer,
                    site: site(&reference.site)?,
                    position: reference.position,
                    sha256,
                },
            )
        }
        Some(VectorSource::Difference {
            minuend,
            subtrahend,
        }) => {
            let (_, a) = read_capture(minuend, base)?;
            let (_, b) = read_capture(subtrahend, base)?;
            if a.len() != b.len() {
                return Err(format!(
                    "the difference's operands have widths {} and {}",
                    a.len(),
                    b.len()
                )
                .into());
            }
            let values: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
            let sha256 = vector_sha256(&values);
            (values, VectorProvenance::Literal { sha256 })
        }
    };
    if let Some(declared) = &entry.sha256 {
        if declared != provenance.sha256() {
            return Err(format!(
                "the declaration claims sha256 {declared} for the vector at layer {} {} but its \
                 bytes hash to {}",
                entry.layer,
                entry.site,
                provenance.sha256()
            )
            .into());
        }
    }
    Ok(match kind {
        InterventionKind::Add => Intervention::add(address, vector, provenance)?,
        InterventionKind::Replace => Intervention::replace(address, vector, provenance)?,
        InterventionKind::Zero => unreachable!("handled above"),
    })
}

/// The run id and the vector a capture file holds at an address, with
/// the stored hash verified against the stored bytes.
fn read_capture(reference: &CapturedRef, base: &Path) -> Result<(String, Vec<f32>), BoxErr> {
    let path = if reference.file.is_absolute() {
        reference.file.clone()
    } else {
        base.join(&reference.file)
    };
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read the capture file {}: {e}", path.display()))?;
    let file: CaptureFile = serde_json::from_str(&text)
        .map_err(|e| format!("the capture file {} does not parse: {e}", path.display()))?;
    let wanted = site(&reference.site)?;
    let entry = file
        .captures
        .iter()
        .find(|c| {
            c.layer == reference.layer
                && c.position == reference.position
                && site(&c.site).ok() == Some(wanted)
        })
        .ok_or_else(|| {
            format!(
                "the capture file {} holds nothing at layer {} {} position {}",
                path.display(),
                reference.layer,
                reference.site,
                reference.position
            )
        })?;
    let actual = vector_sha256(&entry.values);
    if actual != entry.sha256 {
        return Err(format!(
            "the capture file {} is not what it says: the vector at layer {} {} position {} \
             hashes to {actual}, the file claims {}",
            path.display(),
            entry.layer,
            entry.site,
            entry.position,
            entry.sha256
        )
        .into());
    }
    Ok((file.run_id, entry.values.clone()))
}

/// Write every captured address. An address that was declared for
/// capture and never reached is a refusal, not a silently shorter file.
pub fn write_capture(
    path: &Path,
    run_id: &str,
    capture: &CarrierCapture,
    addresses: &[(usize, SublayerSite, usize)],
) -> Result<(), BoxErr> {
    let mut captures = Vec::with_capacity(addresses.len());
    let mut missing = Vec::new();
    for &(layer, site, position) in addresses {
        match capture.get(layer, site, position) {
            Some(values) => captures.push(CaptureEntry {
                layer,
                site: site_name(site).to_string(),
                position,
                sha256: vector_sha256(values),
                values: values.to_vec(),
            }),
            None => missing.push(format!("{layer}:{}:{position}", site_name(site))),
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "capture declared at {} but the run never reached it",
            missing.join(", ")
        )
        .into());
    }
    let file = CaptureFile {
        run_id: run_id.to_string(),
        captures,
    };
    std::fs::write(path, serde_json::to_string(&file)?)?;
    Ok(())
}

/// One observer feeding the recorder and, when armed, a carrier capture.
pub struct Tee<'a, 'b> {
    pub recorder: &'a mut RunRecorder<'b>,
    pub capture: Option<&'a mut CarrierCapture>,
}

impl StepObserver for Tee<'_, '_> {
    fn event(&mut self, event: StepEvent) {
        if let Some(capture) = self.capture.as_mut() {
            capture.event(event.clone());
        }
        self.recorder.event(event);
    }

    fn entering_carrier(&mut self, position: usize, values: &[f32]) {
        if let Some(capture) = self.capture.as_mut() {
            capture.entering_carrier(position, values);
        }
        self.recorder.entering_carrier(position, values);
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        if let Some(capture) = self.capture.as_mut() {
            capture.carrier_write(CarrierWriteRecord {
                layer: record.layer,
                site: record.site,
                position: record.position,
                delta: record.delta,
                after: record.after,
                layer_scale: record.layer_scale,
            });
        }
        self.recorder.carrier_write(record);
    }
}
