//! CONTINUATION-CODEC-3 W1: the freeze as the authority — its subject
//! binding, its bank, its arms — and the verdict routed through the binding.
//!
//! Frozen: docs/represent/forecasts/continuation-codec-3.json. The inherited
//! rule is CODEC-2's (`codec_2::adjudicate`, unchanged); it names its
//! candidate `CH`, and `acceptance.rule_binding` says which arm that symbol
//! means for the verdict. Nothing here hard-codes the verdict subject: it is
//! read from the freeze file, and every arm is adjudicated by the same rule.
//! CODEC-2's and the CODEC-3 reconnaissance's modules are used, not edited.

use std::collections::BTreeMap;

use larql_vindex::format::vindex3::represent::measure::plan::metrics::PositionMetrics;
use sha2::{Digest, Sha256};

use super::codec_2::{adjudicate, BANK_LEN};
use super::*;

const FREEZE_PATH: &str = "../../docs/represent/forecasts/continuation-codec-3.json";
const BANK_PATH: &str = "../../docs/represent/forecasts/continuation-codec-3-token-bank.json";
/// Bank 3's digest as committed with the freeze (sha256 over u32 LE ids);
/// the freeze file must name the same one.
pub(super) const BANK_IDS_SHA256: &str =
    "80d188d3efa8390437f2b8c50c6f104476ff2f49eb64e38eaf9924ba33537098";

/// The KV width of every compressed arm, as frozen.
pub(super) const CODEC_BITS: u8 = 4;

/// What a compressed arm runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Provider {
    /// codec/v1 at `CODEC_BITS`.
    Codec,
    /// codec-recent/v1 at `CODEC_BITS`, the newest `window` K rows exact.
    Recent { window: usize },
}

/// One compressed arm: its name in the freeze and what it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ArmSpec {
    pub name: &'static str,
    pub provider: Provider,
}

impl ArmSpec {
    /// The checkpoint name: the freeze's arm name, lower-cased.
    pub(super) fn key(&self) -> String {
        self.name.to_lowercase()
    }
}

/// The compressed arms the freeze declares, in run order: CH first, so
/// its trace is the reference every CH-recent arm is held to.
pub(super) const ARMS: [ArmSpec; 3] = [
    ArmSpec {
        name: "CH",
        provider: Provider::Codec,
    },
    ArmSpec {
        name: "CH-recent256",
        provider: Provider::Recent { window: 256 },
    },
    ArmSpec {
        name: "CH-recent128",
        provider: Provider::Recent { window: 128 },
    },
];

/// `acceptance.rule_binding`: which arm the inherited symbol means.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Binding {
    pub inherited_symbol: String,
    pub primary: String,
    pub secondary_report_only: String,
    pub control_report_only: String,
}

impl Binding {
    /// Read from a freeze document; every role must be a declared
    /// compressed arm, the roles distinct, and the symbol the candidate the
    /// inherited rule names.
    pub(super) fn from_freeze(freeze: &Value) -> Result<Self, String> {
        let acceptance = &freeze["acceptance"];
        let b = &acceptance["rule_binding"];
        let field = |k: &str| {
            b[k].as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("rule_binding.{k} is missing"))
        };
        let binding = Self {
            inherited_symbol: field("inherited_symbol")?,
            primary: field("primary")?,
            secondary_report_only: field("secondary_report_only")?,
            control_report_only: field("control_report_only")?,
        };
        let subject = acceptance["rule"]["subject"].as_str().unwrap_or_default();
        if !subject.starts_with(&format!(
            "{} is ACCEPTABLE only if",
            binding.inherited_symbol
        )) {
            return Err(format!(
                "inherited symbol `{}` is not the candidate the inherited rule names",
                binding.inherited_symbol
            ));
        }
        let roles = binding.roles();
        for arm in roles {
            if spec(arm).is_none() {
                return Err(format!(
                    "bound arm `{arm}` is not a declared compressed arm"
                ));
            }
            if acceptance["arms"].get(arm).is_none() {
                return Err(format!("bound arm `{arm}` is not an arm of the freeze"));
            }
        }
        if roles[0] == roles[1] || roles[0] == roles[2] || roles[1] == roles[2] {
            return Err("rule_binding roles are not distinct arms".into());
        }
        Ok(binding)
    }

    fn roles(&self) -> [&str; 3] {
        [
            &self.primary,
            &self.secondary_report_only,
            &self.control_report_only,
        ]
    }
}

/// The compressed arm named `name`, if the freeze declares it.
pub(super) fn spec(name: &str) -> Option<ArmSpec> {
    ARMS.iter().copied().find(|a| a.name == name)
}

/// The committed freeze document.
pub(super) fn freeze() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FREEZE_PATH);
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&raw).expect("the freeze is JSON")
}

/// Bank 3, refused unless it is the one committed and the one the freeze
/// names.
pub(super) fn bank3(freeze: &Value) -> Result<Vec<u32>, String> {
    let named = freeze["bank"]["ids_sha256_u32le"]
        .as_str()
        .unwrap_or_default();
    if named != BANK_IDS_SHA256 {
        return Err(format!(
            "the freeze names bank {named}, not {BANK_IDS_SHA256}"
        ));
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(BANK_PATH);
    let raw = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let doc: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    let ids: Vec<u32> = doc["ids"]
        .as_array()
        .ok_or("the bank has no ids")?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or("an id is not a u32")
        })
        .collect::<Result<_, _>>()?;
    let mut hash = Sha256::new();
    for id in &ids {
        hash.update(id.to_le_bytes());
    }
    let digest = format!("{:x}", hash.finalize());
    if ids.len() != BANK_LEN || digest != BANK_IDS_SHA256 {
        return Err(format!(
            "token bank is not the frozen one: {} ids, sha256 {digest}",
            ids.len()
        ));
    }
    Ok(ids)
}

fn refs(v: &[Vec<PositionMetrics>]) -> Vec<&[PositionMetrics]> {
    v.iter().map(Vec::as_slice).collect()
}

/// Every bound arm adjudicated by the inherited rule against C; the
/// verdict is the primary's adjudication and nothing else's.
pub(super) fn adjudicate_bound(
    binding: &Binding,
    c: &[Vec<PositionMetrics>],
    arms: &BTreeMap<String, Vec<Vec<PositionMetrics>>>,
) -> Value {
    let judged = |arm: &str| {
        let strata = arms
            .get(arm)
            .unwrap_or_else(|| panic!("bound arm `{arm}` has no scores"));
        adjudicate(&refs(c), &refs(strata))
    };
    json!({
        "rule_binding": {
            "inherited_symbol": binding.inherited_symbol,
            "primary": binding.primary,
            "secondary_report_only": binding.secondary_report_only,
            "control_report_only": binding.control_report_only,
        },
        "verdict": judged(&binding.primary),
        "reported_not_adjudicated": {
            "secondary": judged(&binding.secondary_report_only),
            "control": judged(&binding.control_report_only),
        },
    })
}
