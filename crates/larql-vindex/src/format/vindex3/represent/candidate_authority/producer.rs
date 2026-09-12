//! Compilation-side collection of actual decisions. The fresh reader
//! never calls this module or resolves the producer's requested map.
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::{digest_range, finish, ExecutableRootBinding};
use crate::error::VindexError;
use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::index::{RepresentationEntry, Vindex3Index};
use crate::format::vindex3::represent::{
    compile::OperandSeal,
    compiler::{read_source_identity, CandidateIndex, SourceDependency},
    map::PrecisionMap,
    plan_roles::PlanRoles,
    policy::classify_in,
    state::{
        ResolvedDecision, ResolvedDecisionVector, ResolvedEncoding, SurfaceTensor, TensorSurface,
    },
};

pub(crate) struct CompilationAuthority {
    index: CandidateIndex,
    surface: TensorSurface,
    decisions: BTreeMap<(String, String), ResolvedEncoding>,
    files: BTreeMap<String, String>,
}

impl CompilationAuthority {
    pub(crate) fn new(
        src: &Path,
        index: &Vindex3Index,
        map: PrecisionMap,
        text: &BTreeSet<String>,
        roles: &PlanRoles,
    ) -> Result<Self, VindexError> {
        let mut tensors = BTreeMap::new();
        let mut files = BTreeMap::new();
        for entry in index.representations.values() {
            let (header, _) = read_segment_header(&src.join(&entry.segment))?;
            files
                .entry(entry.object.clone())
                .or_insert_with(|| entry.segment.clone());
            for t in header.tensors {
                let role = roles
                    .get(&(entry.object.clone(), t.name.clone()))
                    .copied()
                    .filter(|_| text.contains(&entry.object))
                    .unwrap_or_else(|| {
                        classify_in(
                            text.contains(&entry.object),
                            &entry.object,
                            &t.name,
                            &t.shape,
                        )
                    });
                let tensor = SurfaceTensor::new(&entry.object, &t.name, role, t.shape);
                let key = (entry.object.clone(), t.name);
                if let Some(old) = tensors.insert(key, tensor.clone()) {
                    if old != tensor {
                        return Err(VindexError::Parse(format!(
                            "source representations disagree about surface tensor {}/{}",
                            tensor.object, tensor.tensor
                        )));
                    }
                }
            }
        }
        let surface = TensorSurface::new(tensors.into_values())?;
        let decisions = surface
            .entries()
            .iter()
            .map(|t| {
                (
                    (t.object.clone(), t.tensor.clone()),
                    ResolvedEncoding::Source,
                )
            })
            .collect();
        Ok(Self {
            index: CandidateIndex::new(
                &index.model,
                SourceDependency {
                    identity: read_source_identity(src)?,
                    locator_hint: src.display().to_string(),
                },
                "",
                map,
            ),
            surface,
            decisions,
            files,
        })
    }

    pub(crate) fn decided(&mut self, object: &str, tensor: &str, encoding: ResolvedEncoding) {
        self.decisions
            .insert((object.into(), tensor.into()), encoding);
    }

    /// Read the completed segment table and seal precisely the ranges the
    /// writer emitted. Layout-refused and protected tensors remain source.
    pub(crate) fn written(
        &mut self,
        src: &Path,
        out: &Path,
        entry: &RepresentationEntry,
        target_file: &str,
    ) -> Result<(), VindexError> {
        let source_path = src.join(&entry.segment);
        let target_path = out.join(target_file);
        let (source_header, source_start) = read_segment_header(&source_path)?;
        let (target_header, target_start) = read_segment_header(&target_path)?;
        for t in target_header.tensors {
            let Some(ResolvedEncoding::Compiled(encoding)) =
                self.decisions.get(&(entry.object.clone(), t.name.clone()))
            else {
                continue;
            };
            let original = source_header
                .tensors
                .iter()
                .find(|s| s.name == t.name)
                .ok_or_else(|| {
                    VindexError::Parse(format!("compiled tensor {} has no source", t.name))
                })?;
            self.index.ledger.seal(OperandSeal {
                object: entry.object.clone(),
                tensor: t.name,
                encoding: encoding.clone(),
                source_hash: digest_range(
                    &source_path,
                    source_start + original.offset,
                    original.len,
                )?,
                target_hash: digest_range(&target_path, target_start + t.offset, t.len)?,
                target_offset: target_start + t.offset,
                target_len: t.len,
            });
        }
        self.files.insert(entry.object.clone(), target_file.into());
        Ok(())
    }

    pub(crate) fn finish(mut self, out: &Path) -> Result<CandidateIndex, VindexError> {
        let entries = self
            .decisions
            .into_iter()
            .map(|((object, tensor), encoding)| ResolvedDecision {
                object,
                tensor,
                encoding,
            })
            .collect();
        let decisions = ResolvedDecisionVector::from_entries(&self.surface, entries)?;
        // The compiler has already persisted the executable index. Bind
        // its catalogue and declared graph through the existing semantic
        // identity machinery, independently of this candidate's state id.
        let executable_root = ExecutableRootBinding::Vindex3 {
            semantic_digest: read_source_identity(out)?.semantic_digest(),
        };
        finish(
            &mut self.index,
            out,
            self.surface,
            decisions,
            self.files,
            executable_root,
        )?;
        Ok(self.index)
    }
}
