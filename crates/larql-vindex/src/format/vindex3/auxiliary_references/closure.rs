//! **Admitting the whole dependency closure before a payload is opened.**
//!
//! A codec that depends on another represented object cannot be judged
//! one operand at a time: the object it points at may point somewhere
//! else, two owners may share one target, and a table written by hand can
//! describe a cycle. So the closure is walked in full, from metadata
//! alone — the container's tensor table and the registry — and every way
//! it can be wrong is refused *before* the first byte of payload.
//!
//! What is checked, in the order a walk meets it:
//!
//! ```text
//! target absent          the container holds no such tensor
//! provider absent        its label names no registered codec
//! names disagree         the codec requires what the table does not
//!                        provide, or the table provides what it does
//!                        not require (per extent, both ways)
//! target unusable        the OWNING codec judges the target's metadata
//! cycle                  the walk re-enters an operand it is inside
//! ```
//!
//! Two things the walk deliberately does not do. It never reads a
//! payload — every refusal above is a metadata fact, and a closure that
//! had to decode to find a cycle would be useless as an admission. And it
//! decides no extents: a dependency is walked at its own TERMINAL extent,
//! because that is what the container holds and what a later selection
//! can narrow. Choosing a shallower extent for a dependency is a
//! selection question, and it composes with the parent's fidelity —
//! neither of which belongs to admission.

use std::collections::{BTreeMap, BTreeSet};

use super::{OperandAddress, ReferenceTable};
use crate::error::VindexError;
use crate::format::vindex3::represent::codec::{
    admit_auxiliary_names, AuxiliaryMetadata, CodecRegistry, RepresentationExtent,
};

/// What the closure can learn about an operand without opening payload:
/// the label and shape the CONTAINER records for it.
///
/// A trait so the walk can be exercised without a container — and so the
/// container remains the only authority for what it holds.
pub trait ContainerMetadata {
    fn tensor(&self, address: &OperandAddress) -> Option<(String, Vec<usize>)>;
}

/// One operand the closure starts from, at the extent it will be read at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureRoot {
    pub address: OperandAddress,
    pub extent: RepresentationExtent,
}

impl ClosureRoot {
    pub fn new(address: OperandAddress, extent: RepresentationExtent) -> Self {
        Self { address, extent }
    }
}

/// One dependency, resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDependency {
    /// The name the owner's codec declared.
    pub name: String,
    pub target: OperandAddress,
}

/// Every operand the roots depend on, in an order that resolves.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AuxiliaryClosure {
    order: Vec<OperandAddress>,
    dependencies: BTreeMap<OperandAddress, Vec<ResolvedDependency>>,
}

impl AuxiliaryClosure {
    /// Every operand in the closure, DEEPEST FIRST: an operand appears
    /// only after everything it depends on. A loader walking this order
    /// never needs something it has not already resolved.
    pub fn order(&self) -> &[OperandAddress] {
        &self.order
    }

    /// What `owner` depends on, by declared name, in name order.
    pub fn dependencies_of(&self, owner: &OperandAddress) -> &[ResolvedDependency] {
        self.dependencies
            .get(owner)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    pub fn contains(&self, address: &OperandAddress) -> bool {
        self.dependencies.contains_key(address)
    }

    /// How many distinct operands the closure reached — roots included,
    /// each counted ONCE however many owners share it.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

/// Walk the closure of `roots` and refuse every way it can be wrong,
/// from metadata alone.
pub fn admit(
    roots: &[ClosureRoot],
    table: &ReferenceTable,
    registry: &CodecRegistry,
    container: &dyn ContainerMetadata,
) -> Result<AuxiliaryClosure, VindexError> {
    let mut walk = Walk {
        table,
        registry,
        container,
        closure: AuxiliaryClosure::default(),
    };
    for root in roots {
        walk.visit(&root.address, root.extent, &mut Vec::new())?;
    }
    Ok(walk.closure)
}

struct Walk<'a> {
    table: &'a ReferenceTable,
    registry: &'a CodecRegistry,
    container: &'a dyn ContainerMetadata,
    closure: AuxiliaryClosure,
}

impl Walk<'_> {
    fn visit(
        &mut self,
        address: &OperandAddress,
        extent: RepresentationExtent,
        path: &mut Vec<OperandAddress>,
    ) -> Result<(), VindexError> {
        if self.closure.dependencies.contains_key(address) {
            // Already resolved, by this root or another: a shared target
            // is visited once, which is what makes the closure a set and
            // not a traversal count.
            return Ok(());
        }
        if let Some(at) = path.iter().position(|seen| seen == address) {
            return Err(cycle_refusal(&path[at..], address));
        }
        let (label, shape) = self.describe(address)?;
        let codec = self.registry.resolve(&label, &address.tensor)?;
        let provided = self.table.auxiliaries_of(address);
        let provided_names: Vec<&str> = provided.iter().map(|(name, _)| *name).collect();
        admit_auxiliary_names(
            codec.required_auxiliaries(extent),
            &provided_names,
            &label,
            &address.tensor,
            extent,
        )?;

        path.push(address.clone());
        let mut resolved = Vec::with_capacity(provided.len());
        for (name, target) in &provided {
            let (target_label, target_shape) = self.describe(target)?;
            let target_codec = self.registry.resolve(&target_label, &target.tensor)?;
            // The OWNER judges its dependency, from metadata alone.
            codec.validate_auxiliary(
                name,
                &AuxiliaryMetadata {
                    object: target.object.clone(),
                    tensor: target.tensor.clone(),
                    label: target_label,
                    shape: target_shape,
                    identity: Some(target_codec.identity()),
                },
                &shape,
                extent,
                &address.tensor,
            )?;
            // A dependency is walked at its own terminal extent: the
            // container holds all of it, and narrowing is selection's.
            self.visit(target, target_codec.terminal_extent(), path)?;
            resolved.push(ResolvedDependency {
                name: (*name).to_string(),
                target: (*target).clone(),
            });
        }
        path.pop();

        self.closure.dependencies.insert(address.clone(), resolved);
        self.closure.order.push(address.clone());
        Ok(())
    }

    /// The container's own record for an operand, or a refusal naming what
    /// pointed at it.
    fn describe(&self, address: &OperandAddress) -> Result<(String, Vec<usize>), VindexError> {
        self.container.tensor(address).ok_or_else(|| {
            VindexError::Parse(format!(
                "auxiliary closure: {} is referenced and the container holds no such tensor",
                address.describe()
            ))
        })
    }
}

/// A cycle, spelled as the walk met it: the operands in order, closing
/// back on the one it re-entered.
fn cycle_refusal(cycle: &[OperandAddress], closing: &OperandAddress) -> VindexError {
    let mut spelled: Vec<String> = cycle.iter().map(OperandAddress::describe).collect();
    spelled.push(closing.describe());
    VindexError::Parse(format!(
        "auxiliary closure: the declared dependencies form a cycle — {}; a representation cannot \
         depend on itself, directly or through others",
        spelled.join(" → ")
    ))
}

/// Every distinct operand in a set of closures, once each — what a caller
/// asks when it wants the objects rather than the order.
pub fn distinct(closure: &AuxiliaryClosure) -> BTreeSet<&OperandAddress> {
    closure.order.iter().collect()
}
