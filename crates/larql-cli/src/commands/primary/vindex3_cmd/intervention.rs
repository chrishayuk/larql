//! `vindex3 observe --intervene` and `--capture`/`--capture-heads`: the
//! files the verb reads and writes for V3-INTERVENE-1 (carrier
//! addresses) and V3-INTERVENE-2 (head addresses), and the observer that
//! lets one run feed the recorder and both kinds of capture.
//!
//! A DECLARATION names interventions of BOTH kinds in one file. Each
//! entry gives an address — `layer` plus EXACTLY ONE of `site`
//! (`attention`/`ffn`, a carrier address) or `head` (a query head index,
//! a V3-INTERVENE-2 address) — and `positions`, a `kind`, and for the
//! kinds that carry a vector a `vector`: an inline array (a literal), a
//! reference into a capture file (`{file, layer, site|head, position}` —
//! the value another run wrote there, with that run's identity as
//! provenance), or the difference of two such references (`{minuend,
//! subtrahend}`, both the same address kind — a literal whose hash is
//! computed here; the HEAD-1 CARRIED arm is `add` of `h14^B − h14^A`).
//! Carrier kinds are `zero`/`add`/`replace`; head kinds are
//! `zero`/`scale`/`replace` (`scale` takes a `factor`, not a `vector`).
//! An optional `sha256` on a vector entry is CHECKED against the bytes;
//! absent, it is computed and still travels on the receipt.
//!
//! A CAPTURE FILE (`--capture-out`) or HEAD CAPTURE FILE
//! (`--capture-heads-out`) is what those flags write: the run id and,
//! per captured address, the vector and its hash. A declaration that
//! reads one verifies the stored hash before trusting the bytes.

use std::path::{Path, PathBuf};

use larql_inference::vindex3::RunRecorder;
use larql_vindex::format::vindex3::opplan::exec::intervene::{
    vector_sha256, Address, CarrierCapture, Intervention, InterventionKind, InterventionPlan,
    VectorProvenance,
};
use larql_vindex::format::vindex3::opplan::exec::intervene_heads::{
    HeadAddress, HeadCapture, HeadIntervention, HeadInterventionKind, HeadInterventionPlan,
};
use larql_vindex::format::vindex3::opplan::exec::observe::{
    AttentionHeadRecord, CarrierWriteRecord, StepEvent, StepObserver, SublayerSite,
};
use serde::{Deserialize, Serialize};

type BoxErr = Box<dyn std::error::Error>;

/// The declaration file.
#[derive(Debug, Deserialize)]
pub struct Declaration {
    pub interventions: Vec<Declared>,
}

/// One declared intervention, as the file spells it. Exactly one of
/// `site` (carrier, V3-INTERVENE-1) or `head` (V3-INTERVENE-2) must be
/// present.
#[derive(Debug, Deserialize)]
pub struct Declared {
    pub layer: usize,
    #[serde(default)]
    pub site: Option<String>,
    #[serde(default)]
    pub head: Option<usize>,
    pub positions: Vec<usize>,
    pub kind: String,
    /// `scale`'s factor. Present only on a `head` entry of kind `scale`.
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub vector: Option<VectorSource>,
    /// Checked against the vector's bytes when present.
    #[serde(default)]
    pub sha256: Option<String>,
}

/// Where a declared vector comes from. A carrier reference names `site`;
/// a head reference names `head`; untagged deserialisation disambiguates
/// them structurally.
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
    /// A head's `ctx_h` another run captured, by file and address.
    CapturedHead(HeadCapturedRef),
    /// `minuend − subtrahend`, both captured heads.
    DifferenceHead {
        minuend: HeadCapturedRef,
        subtrahend: HeadCapturedRef,
    },
}

/// A reference into a carrier capture file.
#[derive(Debug, Clone, Deserialize)]
pub struct CapturedRef {
    pub file: PathBuf,
    pub layer: usize,
    pub site: String,
    pub position: usize,
}

/// A reference into a head capture file.
#[derive(Debug, Clone, Deserialize)]
pub struct HeadCapturedRef {
    pub file: PathBuf,
    pub layer: usize,
    pub head: usize,
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

/// What `--capture-heads-out` writes.
#[derive(Debug, Serialize, Deserialize)]
pub struct HeadCaptureFile {
    pub run_id: String,
    pub captures: Vec<HeadCaptureEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HeadCaptureEntry {
    pub layer: usize,
    pub head: usize,
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

fn head_kind(name: &str) -> Result<HeadInterventionKind, BoxErr> {
    match name {
        "zero" => Ok(HeadInterventionKind::Zero),
        "scale" => Ok(HeadInterventionKind::Scale),
        "replace" => Ok(HeadInterventionKind::Replace),
        other => Err(format!(
            "unknown head intervention kind `{other}`: expected `zero`, `scale` or `replace`"
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

/// `layer:head:position[,layer:head:position…]`.
pub fn parse_head_addresses(list: &str) -> Result<Vec<(usize, usize, usize)>, BoxErr> {
    list.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|item| {
            let parts: Vec<&str> = item.split(':').collect();
            if parts.len() != 3 {
                return Err(
                    format!("capture address `{item}` is not `layer:head:position`").into(),
                );
            }
            Ok((parts[0].parse()?, parts[1].parse()?, parts[2].parse()?))
        })
        .collect()
}

/// Read a declaration file into a carrier plan and a head plan. Relative
/// capture-file paths resolve against the declaration's own directory.
pub fn load_plan(path: &Path) -> Result<(InterventionPlan, HeadInterventionPlan), BoxErr> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read the declaration {}: {e}", path.display()))?;
    let declaration: Declaration = serde_json::from_str(&text)
        .map_err(|e| format!("the declaration {} does not parse: {e}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut plan = InterventionPlan::none();
    let mut heads = HeadInterventionPlan::none();
    for entry in &declaration.interventions {
        match (&entry.site, entry.head) {
            (Some(_), Some(_)) => {
                return Err(format!(
                    "layer {} declares both `site` and `head`; an entry is one or the other",
                    entry.layer
                )
                .into())
            }
            (None, None) => {
                return Err(
                    format!("layer {} declares neither `site` nor `head`", entry.layer).into(),
                )
            }
            (Some(_), None) => plan = plan.with(resolve(entry, base)?)?,
            (None, Some(_)) => heads = heads.with(resolve_head(entry, base)?)?,
        }
    }
    Ok((plan, heads))
}

fn resolve(entry: &Declared, base: &Path) -> Result<Intervention, BoxErr> {
    let site_name = entry.site.as_deref().expect("caller matched on Some(site)");
    let address = Address::new(
        entry.layer,
        site(site_name)?,
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
            let values = difference(&a, &b)?;
            let sha256 = vector_sha256(&values);
            (values, VectorProvenance::Literal { sha256 })
        }
        Some(VectorSource::CapturedHead(_) | VectorSource::DifferenceHead { .. }) => {
            return Err(format!(
                "layer {} site entry's vector references a head capture; site entries need a \
                 carrier capture (`site`, not `head`, on the reference)",
                entry.layer
            )
            .into())
        }
    };
    if let Some(declared) = &entry.sha256 {
        if declared != provenance.sha256() {
            return Err(format!(
                "the declaration claims sha256 {declared} for the vector at layer {} {} but its \
                 bytes hash to {}",
                entry.layer,
                site_name,
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

fn resolve_head(entry: &Declared, base: &Path) -> Result<HeadIntervention, BoxErr> {
    let head = entry.head.expect("caller matched on Some(head)");
    let address = HeadAddress::new(entry.layer, head, entry.positions.iter().copied())?;
    let kind = head_kind(&entry.kind)?;
    if kind == HeadInterventionKind::Zero {
        if entry.vector.is_some() || entry.factor.is_some() {
            return Err("a `zero` head intervention carries no vector or factor".into());
        }
        return Ok(HeadIntervention::zero(address));
    }
    if kind == HeadInterventionKind::Scale {
        if entry.vector.is_some() {
            return Err("a `scale` head intervention carries a `factor`, not a `vector`".into());
        }
        let factor = entry
            .factor
            .ok_or("a `scale` head intervention needs a `factor`")?;
        return Ok(HeadIntervention::scale(address, factor)?);
    }
    // Replace.
    let (vector, provenance) = match entry.vector.as_ref() {
        None => return Err("a `replace` head intervention needs a `vector`".into()),
        Some(VectorSource::Literal(values)) => {
            let sha256 = vector_sha256(values);
            (values.clone(), VectorProvenance::Literal { sha256 })
        }
        Some(VectorSource::CapturedHead(reference)) => {
            let (run_id, values) = read_head_capture(reference, base)?;
            let sha256 = vector_sha256(&values);
            (
                values,
                VectorProvenance::CapturedHead {
                    run_id,
                    layer: reference.layer,
                    head: reference.head,
                    position: reference.position,
                    sha256,
                },
            )
        }
        Some(VectorSource::DifferenceHead {
            minuend,
            subtrahend,
        }) => {
            let (_, a) = read_head_capture(minuend, base)?;
            let (_, b) = read_head_capture(subtrahend, base)?;
            let values = difference(&a, &b)?;
            let sha256 = vector_sha256(&values);
            (values, VectorProvenance::Literal { sha256 })
        }
        Some(VectorSource::Captured(_) | VectorSource::Difference { .. }) => {
            return Err(format!(
                "layer {} head entry's vector references a carrier capture; head entries need a \
                 head capture (`head`, not `site`, on the reference)",
                entry.layer
            )
            .into())
        }
    };
    if let Some(declared) = &entry.sha256 {
        if declared != provenance.sha256() {
            return Err(format!(
                "the declaration claims sha256 {declared} for the vector at layer {} head {head} \
                 but its bytes hash to {}",
                entry.layer,
                provenance.sha256()
            )
            .into());
        }
    }
    Ok(HeadIntervention::replace(address, vector, provenance)?)
}

fn difference(a: &[f32], b: &[f32]) -> Result<Vec<f32>, BoxErr> {
    if a.len() != b.len() {
        return Err(format!(
            "the difference's operands have widths {} and {}",
            a.len(),
            b.len()
        )
        .into());
    }
    Ok(a.iter().zip(b).map(|(x, y)| x - y).collect())
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

/// The run id and the `ctx_h` a head capture file holds at an address,
/// with the stored hash verified against the stored bytes.
fn read_head_capture(
    reference: &HeadCapturedRef,
    base: &Path,
) -> Result<(String, Vec<f32>), BoxErr> {
    let path = if reference.file.is_absolute() {
        reference.file.clone()
    } else {
        base.join(&reference.file)
    };
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read the head capture file {}: {e}", path.display()))?;
    let file: HeadCaptureFile = serde_json::from_str(&text).map_err(|e| {
        format!(
            "the head capture file {} does not parse: {e}",
            path.display()
        )
    })?;
    let entry = file
        .captures
        .iter()
        .find(|c| {
            c.layer == reference.layer
                && c.head == reference.head
                && c.position == reference.position
        })
        .ok_or_else(|| {
            format!(
                "the head capture file {} holds nothing at layer {} head {} position {}",
                path.display(),
                reference.layer,
                reference.head,
                reference.position
            )
        })?;
    let actual = vector_sha256(&entry.values);
    if actual != entry.sha256 {
        return Err(format!(
            "the head capture file {} is not what it says: the vector at layer {} head {} \
             position {} hashes to {actual}, the file claims {}",
            path.display(),
            entry.layer,
            entry.head,
            entry.position,
            entry.sha256
        )
        .into());
    }
    Ok((file.run_id, entry.values.clone()))
}

/// Write every captured carrier address. An address that was declared
/// for capture and never reached is a refusal, not a silently shorter
/// file.
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

/// Write every captured head address. An address that was declared for
/// capture and never reached is a refusal, not a silently shorter file.
pub fn write_head_capture(
    path: &Path,
    run_id: &str,
    capture: &HeadCapture,
    addresses: &[(usize, usize, usize)],
) -> Result<(), BoxErr> {
    let mut captures = Vec::with_capacity(addresses.len());
    let mut missing = Vec::new();
    for &(layer, head, position) in addresses {
        match capture.get(layer, head, position) {
            Some(values) => captures.push(HeadCaptureEntry {
                layer,
                head,
                position,
                sha256: vector_sha256(values),
                values: values.to_vec(),
            }),
            None => missing.push(format!("{layer}:{head}:{position}")),
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "head capture declared at {} but the run never reached it",
            missing.join(", ")
        )
        .into());
    }
    let file = HeadCaptureFile {
        run_id: run_id.to_string(),
        captures,
    };
    std::fs::write(path, serde_json::to_string(&file)?)?;
    Ok(())
}

/// One observer feeding the recorder and, when armed, a carrier and/or
/// head capture.
pub struct Tee<'a, 'r> {
    pub recorder: &'a mut RunRecorder<'r>,
    pub capture: Option<&'a mut CarrierCapture>,
    pub head_capture: Option<&'a mut HeadCapture>,
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

    // V3-HEAD-OBS-1 rides through the tee: the executor asks the tee, so
    // the tee answers for the recorder and forwards every head record —
    // to the recorder, and, when armed, to the head capture (J7). The
    // head capture sees the UNINTERVENED value: the tap always fires
    // before a V3-INTERVENE-2 head intervention applies (J3).
    fn wants_attention_heads(&self) -> bool {
        self.recorder.wants_attention_heads() || self.head_capture.is_some()
    }

    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        if let Some(capture) = self.head_capture.as_mut() {
            capture.observe(layer, &record);
        }
        self.recorder.attention_head(layer, record);
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
