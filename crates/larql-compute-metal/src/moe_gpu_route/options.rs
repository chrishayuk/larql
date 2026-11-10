//! Operator switches for the routed expert pass, each read once from the
//! environment: the timing-only ablation, the measured-but-not-default
//! kernel arms, and VERIFY-N's expert sharing. Each switch is a pure parser
//! over the variable's value plus a read-once getter, so every arm of the
//! parser is testable without touching the process environment.

use std::sync::OnceLock;

/// Timing-only ablation of the expert tail machinery, for the A-12
/// in-situ decomposition: `LARQL_ABLATE_MOE=bias,act,combine` (any
/// subset). **Numbers are wrong by construction** — the run exists to
/// price a component's wall time, never to produce output — and the
/// switch announces itself once on stderr so no measurement can quietly
/// inherit it.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(super) struct MoeAblation {
    pub(super) bias: bool,
    pub(super) act: bool,
    pub(super) combine: bool,
}

/// The ablation a `LARQL_ABLATE_MOE` value names, and the parts it did not
/// recognise.
fn parse_moe_ablation(spec: Option<&str>) -> (MoeAblation, Vec<String>) {
    let mut a = MoeAblation::default();
    let mut unknown = Vec::new();
    for part in spec.unwrap_or("").split(',') {
        match part.trim() {
            "bias" => a.bias = true,
            "act" => a.act = true,
            "combine" => a.combine = true,
            "" => {}
            other => unknown.push(other.to_string()),
        }
    }
    (a, unknown)
}

/// The stderr announcement an active ablation makes; `None` when inactive.
fn ablation_banner(a: MoeAblation) -> Option<String> {
    (a.bias || a.act || a.combine).then(|| {
        format!(
            "[moe] ABLATED RUN — skipping{}{}{}; numbers are wrong by construction, timing only",
            if a.bias { " bias" } else { "" },
            if a.act { " act" } else { "" },
            if a.combine { " combine" } else { "" },
        )
    })
}

pub(super) fn moe_ablation() -> MoeAblation {
    static A: OnceLock<MoeAblation> = OnceLock::new();
    *A.get_or_init(|| {
        let (a, unknown) = parse_moe_ablation(std::env::var("LARQL_ABLATE_MOE").ok().as_deref());
        for part in unknown {
            eprintln!("[moe] unknown LARQL_ABLATE_MOE part: {part}");
        }
        if let Some(banner) = ablation_banner(a) {
            eprintln!("{banner}");
        }
        a
    })
}

/// A switch that is on only when its variable is exactly `"1"`.
fn opt_in(value: Option<&str>) -> bool {
    value == Some("1")
}

/// A switch that is on unless its variable is exactly `"0"`.
fn opt_out(value: Option<&str>) -> bool {
    value != Some("0")
}

fn read_once(cell: &'static OnceLock<bool>, var: &str, parse: fn(Option<&str>) -> bool) -> bool {
    *cell.get_or_init(|| parse(std::env::var(var).ok().as_deref()))
}

/// Opt-in for the fused down+combine dispatch (`LARQL_MXFP4_EXPERT_DC=1`).
/// A/B/A/B on gpt-oss read −0.21 then +0.12 ms under battery drift —
/// ambiguous, so the split form stays the default until a rested AC
/// re-run decides. (The in-situ ablation's −0.32 ms for `combine` was
/// mostly a dependency-break artifact: skipping the kernel also freed
/// the next layer from waiting on the down projection.)
pub(super) fn expert_down_combine_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    read_once(&ON, "LARQL_MXFP4_EXPERT_DC", opt_in)
}

/// Control for the x2 expert GEMV arm: `LARQL_MXFP4_EXPERT_X2=0` keeps
/// the single-row vec kernel. Read once.
pub(super) fn expert_x2_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    read_once(&ON, "LARQL_MXFP4_EXPERT_X2", opt_out)
}

/// Opt-in for the fused gate+up expert dispatch (`LARQL_MXFP4_EXPERT_GU=1`).
/// Measured NULL on gpt-oss (−0.04 ms, within noise): the second GEMV's α
/// is hidden behind the first's tail and 1440 threadgroups already fill
/// the machine — the class-aware α model's prediction. Retained as an arm.
pub(super) fn expert_gu_fusion_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    read_once(&ON, "LARQL_MXFP4_EXPERT_GU", opt_in)
}

/// Operator control for VERIFY-N rung 5b — how a block's expert pass
/// shares a distinct expert's reads: `LARQL_ROUTED_DEDUPE=sorted`
/// (default: every slot its own threadgroup, slots dispatched in expert
/// order so shared tiles come from cache), `=multi` (one threadgroup per
/// distinct expert, looping its slots), `=0` (neither — rung 5a, the A
/// arm). Read once.
pub const ROUTED_DEDUPE_ENV: &str = "LARQL_ROUTED_DEDUPE";

/// How a block's expert pass shares a distinct expert's reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DedupeArm {
    Sorted,
    Multi,
}

/// The arm a `LARQL_ROUTED_DEDUPE` value names (`None` = off); an
/// unknown value is an error naming it, never a silent default.
fn parse_dedupe_arm(value: Option<&str>) -> Result<Option<DedupeArm>, String> {
    match value {
        Some("0") => Ok(None),
        Some("multi") => Ok(Some(DedupeArm::Multi)),
        Some("sorted") | None => Ok(Some(DedupeArm::Sorted)),
        Some(other) => Err(format!("{ROUTED_DEDUPE_ENV}: unknown arm {other:?}")),
    }
}

pub(super) fn routed_dedupe_arm() -> Option<DedupeArm> {
    static ARM: OnceLock<Option<DedupeArm>> = OnceLock::new();
    *ARM.get_or_init(|| {
        parse_dedupe_arm(std::env::var(ROUTED_DEDUPE_ENV).ok().as_deref())
            .unwrap_or_else(|why| panic!("{why}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ablation_parts_parse_and_unknown_parts_are_named() {
        assert_eq!(parse_moe_ablation(None), (MoeAblation::default(), vec![]));
        let (a, unknown) = parse_moe_ablation(Some("bias, act ,combine,,nope"));
        assert_eq!(
            a,
            MoeAblation {
                bias: true,
                act: true,
                combine: true
            }
        );
        assert_eq!(unknown, vec!["nope".to_string()]);
    }

    #[test]
    fn only_an_active_ablation_announces_itself_naming_its_parts() {
        assert_eq!(ablation_banner(MoeAblation::default()), None);
        let banner = ablation_banner(MoeAblation {
            bias: false,
            act: true,
            combine: true,
        })
        .expect("active");
        assert!(banner.contains("skipping act combine"), "{banner}");
    }

    #[test]
    fn opt_in_and_opt_out_switches_read_exactly_one_value() {
        assert!(opt_in(Some("1")));
        assert!(!opt_in(None) && !opt_in(Some("0")) && !opt_in(Some("true")));
        assert!(opt_out(None) && opt_out(Some("1")));
        assert!(!opt_out(Some("0")));
    }

    #[test]
    fn dedupe_arms_parse_and_an_unknown_arm_is_refused_by_name() {
        assert_eq!(parse_dedupe_arm(None), Ok(Some(DedupeArm::Sorted)));
        assert_eq!(
            parse_dedupe_arm(Some("sorted")),
            Ok(Some(DedupeArm::Sorted))
        );
        assert_eq!(parse_dedupe_arm(Some("multi")), Ok(Some(DedupeArm::Multi)));
        assert_eq!(parse_dedupe_arm(Some("0")), Ok(None));
        let why = parse_dedupe_arm(Some("fast")).expect_err("unknown arm");
        assert!(why.contains("\"fast\""), "{why}");
    }

    #[test]
    fn the_read_once_getters_resolve_in_this_process() {
        // Whatever the environment says, each getter resolves (and the
        // second call returns the cached first answer).
        assert_eq!(moe_ablation(), moe_ablation());
        assert_eq!(expert_x2_enabled(), expert_x2_enabled());
        assert_eq!(expert_down_combine_enabled(), expert_down_combine_enabled());
        assert_eq!(expert_gu_fusion_enabled(), expert_gu_fusion_enabled());
        assert_eq!(routed_dedupe_arm(), routed_dedupe_arm());
    }
}
