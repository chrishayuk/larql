//! `check_ledger` for an operation target (`weight-patches/v4` §6): consistency of the record with the container
//! and with itself. It does not recompute any op; an independent verifier does that (§7).

use std::collections::BTreeSet;
use std::path::Path;

use super::any_op::WeightOpRef;
use super::ledger::{LedgerSource, LedgerTarget};
use super::op_target::{
    all_zero_valued, check_cost, cost_of, tensor_sha256, StructuredCost, StructuredOpRecord,
};
use crate::format::vindex3::encode::segment::{read_exact_at, read_segment_header};

const BF16_DTYPE: &str = "BF16";
const BF16_BYTES: usize = 2;

/// Defects of one operation target against the canonical segment at `segment` and the ledger's `source`.
pub(super) fn check_operation_target(
    root: &Path,
    segment: &str,
    source: &LedgerSource,
    target: &LedgerTarget,
) -> Vec<String> {
    let name = &target.tensor;
    let mut defects = Vec::new();
    if !target.edits.is_empty() {
        defects.push(format!(
            "ledger target `{name}` lists both element edits and operations"
        ));
    }
    let (Some(base_sha), Some(result_sha), Some(changed)) = (
        target.base_tensor_sha256.as_deref(),
        target.result_tensor_sha256.as_deref(),
        target.changed_elements,
    ) else {
        defects.push(format!(
            "operation target `{name}` lacks its base or result digest or its changed-element count"
        ));
        return defects;
    };
    let [rows, cols] = match target.shape.as_slice() {
        [r, c] if target.dtype == BF16_DTYPE => [*r, *c],
        _ => {
            defects.push(format!("operation target `{name}` is not a BF16 matrix"));
            return defects;
        }
    };
    match stored_tensor(root, segment, name, &target.shape) {
        Ok(bits) if tensor_sha256(&bits) == result_sha => {}
        Ok(_) => defects.push(format!(
            "`{name}` does not hash to the ledger's result_tensor_sha256"
        )),
        Err(e) => defects.push(e),
    }
    // A sham (every op zero-valued) records no change and its result is its base; anything else records one.
    let sham = all_zero_valued(&target.operations).unwrap_or(false);
    if sham && (base_sha != result_sha || changed != 0) {
        defects.push(format!(
            "operation target `{name}` holds only zero-valued ops but records a change to its tensor"
        ));
    }
    if !sham && (base_sha == result_sha || changed == 0) {
        defects.push(format!(
            "operation target `{name}` records no change to its tensor"
        ));
    }
    let mut written = BTreeSet::new();
    let mut total = 0usize;
    for (i, entry) in target.operations.iter().enumerate() {
        let op = &entry.op;
        let at = format!("operation {i} of `{name}`");
        if let Err(e) = op.as_ref().check_intact() {
            defects.push(format!("{at}: {e}"));
            continue;
        }
        if op.as_ref().tensor() != name {
            defects.push(format!("{at} names tensor `{}`", op.as_ref().tensor()));
        }
        let bound = op.source();
        if bound.index_sha256 != source.index_sha256
            || bound.representation != target.representation
            || bound.payload_sha256 != source.canonical.payload_sha256
        {
            defects.push(format!("{at} is bound to another source than the ledger's"));
        }
        match op.operands_sha256() {
            Ok(digests) if digests == entry.operands_sha256 => {}
            Ok(_) => defects.push(format!("{at}: operands_sha256 does not match its operands")),
            Err(e) => defects.push(format!("{at}: {e}")),
        }
        for &c in op.as_ref().structured_columns().unwrap_or(&[]) {
            if c >= cols {
                defects.push(format!("{at} writes column {c} of a {cols}-column tensor"));
            }
            if !written.insert(c) {
                defects.push(format!(
                    "{at} writes column {c}, which another operation also writes"
                ));
            }
        }
        match (op.is_zero_valued(), entry.changed_elements) {
            (Err(e), _) => defects.push(format!("{at}: {e}")),
            (Ok(false), 0) => defects.push(format!("{at} records no changed element")),
            (Ok(true), n) if n > 0 => defects.push(format!(
                "{at} is zero-valued but records {n} changed element(s)"
            )),
            _ => {}
        }
        total = total.saturating_add(entry.changed_elements);
    }
    if total != changed {
        defects.push(format!(
            "operation target `{name}` records {changed} changed elements; its operations add up to {total}"
        ));
    }
    let refs: Vec<WeightOpRef<'_>> = target
        .operations
        .iter()
        .map(|o| StructuredOpRecord::as_ref(&o.op))
        .collect();
    if let Err(e) = check_cost(&StructuredCost {
        tensor_elements: (rows as u64).saturating_mul(cols as u64),
        ..cost_of(&refs)
    }) {
        defects.push(format!("operation target `{name}`: {e}"));
    }
    defects
}

/// The stored BF16 bits of `name` in the segment at `segment`, checked against the ledger's shape.
fn stored_tensor(
    root: &Path,
    segment: &str,
    name: &str,
    shape: &[usize],
) -> Result<Vec<u16>, String> {
    let path = root.join(segment);
    let (header, payload_start) = read_segment_header(&path)
        .map_err(|e| format!("cannot read `{segment}` to check the ledger: {e}"))?;
    let tensor = header
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("ledger edits tensor `{name}`, which `{segment}` does not hold"))?;
    if tensor.dtype != BF16_DTYPE || tensor.shape != shape {
        return Err(format!(
            "ledger target `{name}` dtype or shape differs from the segment's tensor table"
        ));
    }
    let file = std::fs::File::open(&path).map_err(|e| format!("cannot open `{segment}`: {e}"))?;
    let mut bytes = vec![0u8; tensor.len as usize];
    read_exact_at(&file, &mut bytes, payload_start + tensor.offset)
        .map_err(|e| format!("cannot read `{name}`: {e}"))?;
    Ok(bytes
        .chunks_exact(BF16_BYTES)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}
