//! The precision map — a compiled program saying what encoding each
//! tensor is in, and the authority every consumer answers to.
//!
//! ```text
//! model semantics
//!       ↓
//! precision map          ← authority
//!       ├───────────────┐
//!       ↓               ↓
//! stored pack        transient encoder
//!       ↓               ↓
//!              backend  ← proves it can execute the formats the map chose
//! ```
//!
//! Before this existed the dependency ran the other way: the pack carried
//! both the bytes and, implicitly, the decisions, and the transient oracle
//! recovered the decisions by reading the pack's tensor table. That was
//! *operationally* correct — both arms ran the same program — but it made
//! the compiled artifact the authority for its own correctness, which is
//! the one thing an artifact cannot be. There was no way to ask why a
//! tensor is BF16 except by observing that some pack stored it that way.
//!
//! With the map explicit:
//!
//! - **stored** checks its pack *conforms* to the map, rather than
//!   defining it;
//! - **transient** manufactures exactly what the map says is represented,
//!   reading no pack at all;
//! - **auto** uses stored bytes where they exist and manufactures the rest
//!   of what the map requires;
//! - the backend proves only that it can execute the formats the map
//!   selected.
//!
//! ## Structural, not enumerated
//!
//! A Glimmer stack is 416 compiled tensors and a Granite stack 280. Listing
//! them would make the map a transcript of a particular model rather than a
//! policy, and two containers running the same policy would carry
//! different-looking maps. So the map is rules: a default encoding, the
//! roles it applies to, and ordered exceptions. `r1-protect-v` is four
//! lines whatever it is compiled against.

use serde::{Deserialize, Serialize};

use super::policy::{layer_of, projection_of, Protections, Role, RolePolicy};

/// What a map decides for one tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Precision<'a> {
    /// Compile it to this encoding.
    Compiled(&'a str),
    /// Hold it at whatever precision the source representation has.
    Source,
}

/// One exception to the default, matched in declaration order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Exception {
    /// Projection this applies to, e.g. `v_proj`. `None` matches any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<String>,
    /// Inclusive depth range. `None` matches any depth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<(u32, u32)>,
    /// What matching tensors get. `None` = source precision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding: Option<String>,
}

impl Exception {
    /// Whether this exception governs `tensor`.
    ///
    /// An exception with neither a projection nor a range matches
    /// everything, which is a legitimate way to write "compile nothing".
    pub(super) fn matches(&self, tensor: &str) -> bool {
        if let Some(p) = &self.projection {
            if projection_of(tensor) != Some(p.as_str()) {
                return false;
            }
        }
        if let Some((lo, hi)) = self.layers {
            match layer_of(tensor) {
                Some(l) if l >= lo && l <= hi => {}
                // A tensor with no depth cannot be inside a depth range.
                _ => return false,
            }
        }
        true
    }

    pub(super) fn describe(&self) -> String {
        let mut s = self.projection.clone().unwrap_or_else(|| "*".into());
        if let Some((lo, hi)) = self.layers {
            s.push_str(&format!(" layers {lo}-{hi}"));
        }
        s.push_str(" -> ");
        s.push_str(self.encoding.as_deref().unwrap_or("source"));
        s
    }
}

/// The compiled precision program.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrecisionMap {
    /// Identity, so a container names the program that produced it and two
    /// artifacts can be compared by policy rather than by bytes.
    pub name: String,
    /// Encoding an eligible tensor takes unless an exception says
    /// otherwise.
    pub encoding: String,
    /// Roles the program compiles at all. Everything else is source
    /// precision, and unnamed roles stay source precision — the same
    /// fail-safe the role policy has.
    pub roles: Vec<String>,
    /// Ordered exceptions; the first match decides.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exceptions: Vec<Exception>,
}

impl PrecisionMap {
    /// Build the map a compilation is about to perform.
    pub fn from_policy(
        name: impl Into<String>,
        encoding: impl Into<String>,
        roles: &RolePolicy,
        protect: &Protections,
    ) -> Self {
        Self {
            name: name.into(),
            encoding: encoding.into(),
            roles: roles.roles().iter().map(|r| r.name().to_string()).collect(),
            exceptions: protect.as_exceptions(),
        }
    }

    /// What this map decides for a tensor of `role`.
    ///
    /// Role first: a map never compiles a role it does not name, whatever
    /// the exceptions say, so an exception cannot widen eligibility by
    /// accident.
    pub fn resolve(&self, role: Role, tensor: &str) -> Precision<'_> {
        if !self.roles.iter().any(|r| r == role.name()) {
            return Precision::Source;
        }
        for e in &self.exceptions {
            if e.matches(tensor) {
                return match &e.encoding {
                    Some(enc) => Precision::Compiled(enc),
                    None => Precision::Source,
                };
            }
        }
        Precision::Compiled(&self.encoding)
    }

    /// Whether a tensor stored as `dtype` conforms to what this map says.
    ///
    /// The check `stored` owes: a pack is a claim about bytes, and the map
    /// is the authority those bytes are supposed to satisfy. A pack that
    /// disagrees is not a pack for this program.
    pub fn conforms(&self, role: Role, tensor: &str, stored_dtype: &str) -> bool {
        match self.resolve(role, tensor) {
            Precision::Compiled(enc) => stored_dtype == enc,
            // Source precision is whatever the checkpoint had, so any
            // non-compiled dtype satisfies it; only claiming the compiled
            // encoding would be a contradiction.
            Precision::Source => stored_dtype != self.encoding,
        }
    }

    /// Human-readable program, for reports and provenance.
    pub fn describe(&self) -> String {
        let mut lines = vec![format!("{} ({} by default)", self.name, self.encoding)];
        lines.push(format!("  roles: {}", self.roles.join(", ")));
        for e in &self.exceptions {
            lines.push(format!("  except {}", e.describe()));
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_with(exceptions: Vec<Exception>) -> PrecisionMap {
        PrecisionMap {
            name: "test".into(),
            encoding: "NVFP4".into(),
            roles: vec!["decoder-linear".into(), "expert-weight".into()],
            exceptions,
        }
    }

    #[test]
    fn an_eligible_role_compiles_by_default() {
        let m = map_with(vec![]);
        assert_eq!(
            m.resolve(Role::DecoderLinear, "0.self_attn.q_proj.weight"),
            Precision::Compiled("NVFP4")
        );
    }

    #[test]
    fn an_unnamed_role_is_source_precision_whatever_the_exceptions_say() {
        // Exceptions narrow; they must never widen. A map that compiled the
        // embedding because an exception happened to match its name would
        // defeat the conservative default entirely.
        let m = map_with(vec![Exception {
            projection: None,
            layers: None,
            encoding: Some("NVFP4".into()),
        }]);
        assert_eq!(m.resolve(Role::Embedding, "weight"), Precision::Source);
        assert_eq!(
            m.resolve(Role::Router, "0.router.weight"),
            Precision::Source
        );
    }

    #[test]
    fn the_first_matching_exception_decides() {
        let m = map_with(vec![
            Exception {
                projection: Some("v_proj".into()),
                layers: Some((0, 9)),
                encoding: None,
            },
            Exception {
                projection: Some("v_proj".into()),
                layers: None,
                encoding: Some("NVFP4".into()),
            },
        ]);
        // Early v_proj hits the first rule.
        assert_eq!(
            m.resolve(Role::DecoderLinear, "3.self_attn.v_proj.weight"),
            Precision::Source
        );
        // Late v_proj falls through to the second.
        assert_eq!(
            m.resolve(Role::DecoderLinear, "30.self_attn.v_proj.weight"),
            Precision::Compiled("NVFP4")
        );
    }

    #[test]
    fn a_depth_range_cannot_claim_a_tensor_with_no_depth() {
        let m = map_with(vec![Exception {
            projection: None,
            layers: Some((0, 100)),
            encoding: None,
        }]);
        // `weight` belongs to no layer; the range says nothing about it.
        assert_eq!(
            m.resolve(Role::DecoderLinear, "weight"),
            Precision::Compiled("NVFP4")
        );
    }

    #[test]
    fn conformance_checks_a_pack_against_the_program() {
        let m = map_with(vec![Exception {
            projection: Some("q_proj".into()),
            layers: None,
            encoding: None,
        }]);
        let q = "0.self_attn.q_proj.weight";
        let k = "0.self_attn.k_proj.weight";

        assert!(m.conforms(Role::DecoderLinear, k, "NVFP4"));
        assert!(
            !m.conforms(Role::DecoderLinear, k, "BF16"),
            "k must be compiled"
        );

        assert!(m.conforms(Role::DecoderLinear, q, "BF16"));
        assert!(
            !m.conforms(Role::DecoderLinear, q, "NVFP4"),
            "a pack that compiled a protected tensor does not satisfy this map"
        );
    }

    #[test]
    fn a_map_is_a_policy_not_a_transcript() {
        // Four lines, whatever it is compiled against — so two containers
        // running the same policy carry the same map.
        let m = PrecisionMap {
            name: "r1-protect-v".into(),
            encoding: "NVFP4".into(),
            roles: vec!["decoder-linear".into()],
            exceptions: vec![Exception {
                projection: Some("v_proj".into()),
                layers: None,
                encoding: None,
            }],
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(
            json.len() < 200,
            "a map must not scale with tensor count: {json}"
        );
        let back: PrecisionMap = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
        assert!(m.describe().contains("v_proj -> source"));
    }
}
