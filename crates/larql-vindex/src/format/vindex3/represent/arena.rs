//! **Alternative physical representations, materialised on demand from
//! the container.**
//!
//! A quality experiment does not own model weights. It asks the VINDEX
//! for the physical representation each arm requires, and the only thing
//! that differs between arms is the [`PrecisionMap`]:
//!
//! ```text
//! baseline   PrecisionMap = source everywhere
//! candidate  PrecisionMap = source + Exception(RoleScope, Q6_K)
//! ```
//!
//! Both arms therefore execute the SAME semantic model. That is what
//! makes the comparison attributable, and it is why the candidate must
//! not be a second copy of the checkpoint: duplicating 95 GB to test one
//! `RoleScope` would create a second artifact with its own authority,
//! its own drift, and its own opportunity to disagree with the model it
//! claims to represent.
//!
//! ## Keyed by identity, never by address
//!
//! The cache key is `(object, tensor, encoding)` — the operand's
//! semantic identity plus what it was encoded as. Not a pointer, not an
//! offset, not a slot index. Those change when a container is repacked
//! and are shared between unrelated allocations; this codebase has
//! already been bitten three times by a cache keyed on `(ptr, len)`
//! returning a previous tenant's bytes.
//!
//! ## Any expert must be addressable, including one the baseline never
//! routed to
//!
//! If the candidate representation moves a routing decision from expert
//! 73 to expert 181, expert 181 has to resolve and execute. That event
//! is precisely what the quality bank exists to observe. A pre-exported
//! union of the baseline's experts would turn it into a residency
//! refusal instead — fixture construction silently contaminating the
//! measurement it was built to take. Backed by the whole container,
//! every expert is addressable and the question does not arise.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// An operand's cache identity: `(object, tensor, encoding)`.
///
/// Named so the key's SHAPE is visible at every use — it is the whole
/// point of the cache that it is keyed by semantic identity and not by
/// an address.
type OperandKey = (String, String, String);

use super::map::{Precision, PrecisionMap};
use super::policy::Role;
use crate::error::VindexError;
use crate::format::vindex3::opplan::OperandRef;

/// The stored bytes of one operand, exactly as the container holds them.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredOperand {
    pub dtype: String,
    pub bytes: Vec<u8>,
}

/// Where source operands come from.
///
/// A trait rather than a concrete `OperandStore` so the arena's own
/// behaviour — caching, scoping, fail-closed encoding — is testable
/// without a 92 GB container, and so a future source (a remote
/// container, a partially-materialised one) needs no change here.
pub trait SourceOperands {
    fn load_stored(&self, operand: &OperandRef) -> Result<StoredOperand, VindexError>;
}

/// What an arm should execute for one operand.
#[derive(Debug, Clone, PartialEq)]
pub struct Materialised {
    /// The encoding actually produced — the source dtype when the map
    /// left this operand alone.
    pub encoding: String,
    pub bytes: Arc<Vec<u8>>,
    /// Whether this came from the cache rather than being encoded now.
    pub cached: bool,
}

/// Lazily materialised representations under one precision map.
///
/// One arena per ARM. Two arms differing only in their map is the whole
/// experiment; sharing an arena between them would let one arm serve the
/// other's bytes.
pub struct RepresentationArena {
    map: PrecisionMap,
    cache: Mutex<HashMap<OperandKey, Arc<Vec<u8>>>>,
}

impl RepresentationArena {
    pub fn new(map: PrecisionMap) -> Self {
        Self {
            map,
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn map(&self) -> &PrecisionMap {
        &self.map
    }

    /// How many distinct operands this arm has materialised.
    ///
    /// The candidate's arena grows to exactly the scope under test —
    /// `ExpertWeight/down_proj/layers 20..26` materialises those and
    /// nothing else — so this is a direct check that the map governed
    /// what it claimed to.
    pub fn materialised(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// The bytes this arm executes for `operand`.
    ///
    /// Source precision returns the container's own bytes untouched, so
    /// a baseline arm is bit-identical to reading the container
    /// directly — there is no re-encoding round trip to perturb it.
    pub fn resolve(
        &self,
        source: &dyn SourceOperands,
        role: Role,
        operand: &OperandRef,
    ) -> Result<Materialised, VindexError> {
        let encoding = match self.map.resolve(role, &operand.tensor) {
            Precision::Source => {
                let stored = source.load_stored(operand)?;
                return Ok(Materialised {
                    encoding: stored.dtype,
                    bytes: Arc::new(stored.bytes),
                    cached: false,
                });
            }
            Precision::Compiled(enc) => enc.to_string(),
        };

        let key: OperandKey = (
            operand.object.clone(),
            operand.tensor.clone(),
            encoding.clone(),
        );
        if let Some(hit) = self.cache.lock().unwrap().get(&key) {
            return Ok(Materialised {
                encoding,
                bytes: hit.clone(),
                cached: true,
            });
        }

        let stored = source.load_stored(operand)?;
        let values = super::super::opplan::exec::operands::widen(
            &stored.dtype,
            &stored.bytes,
            &operand.tensor,
        )?;
        let bytes = Arc::new(encode(&encoding, &values, &operand.tensor)?);
        self.cache.lock().unwrap().insert(key, bytes.clone());
        Ok(Materialised {
            encoding,
            bytes,
            cached: false,
        })
    }
}

/// Encode widened values into a named representation.
///
/// **Fail-closed.** An unknown encoding is refused rather than passed
/// through as source bytes: silently executing BF16 while a record
/// claims Q6_K would make the whole evidence chain a lie, and the
/// failure would look like "quantisation is free".
fn encode(encoding: &str, values: &[f32], name: &str) -> Result<Vec<u8>, VindexError> {
    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
    match encoding {
        "Q6_K" | "Q4_K" => {
            if !values.len().is_multiple_of(256) {
                return Err(VindexError::Parse(format!(
                    "tensor `{name}`: {} values is not a whole number of 256-element \
                     superblocks, so {encoding} rows would share a scale",
                    values.len()
                )));
            }
            Ok(if encoding == "Q6_K" {
                quantize_q6_k(values)
            } else {
                quantize_q4_k(values)
            })
        }
        "BF16" => Ok(values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()),
        other => Err(VindexError::Parse(format!(
            "tensor `{name}`: no encoder for `{other}` — refusing rather than binding \
             source bytes under a name that claims otherwise"
        ))),
    }
}

#[cfg(test)]
#[path = "arena_tests.rs"]
mod tests;
