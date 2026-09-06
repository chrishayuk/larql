//! Rendering for `larql k3-ledger` — I/O, excluded from coverage.
//!
//! Every quoted ratio names its convention (R0). Anything resting on an
//! unmeasured input says so in the output rather than in a footnote.

use serde_json::json;

use super::args::{BlockArgs, BudgetArgs, FrontierArgs, TouchArgs};
use super::block::{self, DraftProfile, PhysicalReuse, StateTraffic};
use super::budget::{self, LinkPremises};
use super::fetch;
use super::frontier::{self, ServingPremises};
use super::geometry::{BitConvention, K3Geometry};
use super::scenario::{self, ScenarioPremises};
use super::touch::{self, SliceTier};

type R = Result<(), Box<dyn std::error::Error>>;

fn header(geom: &K3Geometry) {
    println!(
        "K3 geometry: {} layers ({} KDA + {} MLA), {}-of-{} ({:.2}% activation), \
         expert {:.2} MB",
        geom.n_layers,
        geom.n_kda_layers,
        geom.n_mla_layers,
        geom.top_k,
        geom.n_experts,
        100.0 * geom.activation_fraction(),
        geom.expert_bytes as f64 / 1e6,
    );
    println!(
        "activated {:.2} B params (dense {:.2} B + routed {:.2} B); vision {}",
        geom.activated_params() as f64 / 1e9,
        geom.dense_params() as f64 / 1e9,
        geom.routed_activated_params() as f64 / 1e9,
        if geom.vision_included {
            "included"
        } else {
            "EXCLUDED"
        },
    );
    if geom.branches_are_equal() {
        println!(
            "branches w1/w2/w3 equal to the byte -> dropping one caps at {:.2}x (retention {:.3})",
            1.0 / geom.up_fold_retention(),
            geom.up_fold_retention(),
        );
    }
}

pub fn budget(geom: &K3Geometry, a: &BudgetArgs, as_json: bool) -> R {
    let link = LinkPremises {
        gb_s: a.link_gb_s,
        target_tok_s: a.target_tok_s,
        ports: a.ports,
    };
    let m = budget::miss_budget(geom, &link);
    let g = budget::granularity(geom, &link, a.feature_fraction, 2.0, a.locality);

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({"budget": m, "granularity": g}))?
        );
        return Ok(());
    }
    header(geom);
    println!();
    println!(
        "miss budget   {:.1} MB/token ({} GB/s x {} port(s) / {} tok/s)",
        m.miss_budget_bytes / 1e6,
        a.link_gb_s,
        a.ports,
        a.target_tok_s
    );
    println!("required      {:.2} GB/token", m.required_fetch_bytes / 1e9);
    println!("GAP           {:.1}x", m.gap);
    println!(
        "  {} expert-visits/token, {:.1} KB allowed at each = {:.3}% of an expert",
        m.expert_visits,
        m.bytes_allowed_per_visit / 1e3,
        100.0 * m.fraction_of_expert_allowed,
    );
    println!();
    println!(
        "--- read granularity at {:.1}% of features ---",
        100.0 * a.feature_fraction
    );
    println!(
        "  feature row {:.0} B, {:.0} rows/visit",
        g.feature_row_bytes, g.features_selected
    );
    println!(
        "  ideal {:.0} KB/visit -> paged {:.2} MB/visit ({:.2}x amplification)",
        g.ideal_bytes_per_visit / 1e3,
        g.paged_bytes_per_visit / 1e6,
        g.read_amplification
    );
    println!(
        "  {:.2}M IOPS required vs {:.2}M bandwidth-equivalent -> {:.1}x short",
        g.iops_required / 1e6,
        g.iops_bandwidth_equivalent / 1e6,
        g.iops_shortfall
    );
    println!("  (request rate binds before bandwidth does)");
    Ok(())
}

pub fn touch(geom: &K3Geometry, p: &ServingPremises, a: &TouchArgs, as_json: bool) -> R {
    let l = touch::touch_ledger(geom, p, a.target_tok_s);
    let slices: Vec<_> = a
        .slice_gb
        .iter()
        .map(|gb| {
            touch::slice_composition(
                geom,
                p,
                &SliceTier {
                    size_bytes: gb * 1e9,
                },
            )
        })
        .collect();

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({"ledger": l, "slices": slices}))?
        );
        return Ok(());
    }
    header(geom);
    println!();
    println!(
        "--- per-token touch: dense {:.2} all-in bits [{}] ---",
        p.dense_all_in_bits(geom),
        p.dense.label(geom)
    );
    println!(
        // NOT "irreducible": this 111 GB is exactly what REPRESENT exists to
        // reduce. What is irreducible is that these operations are ACTIVATED
        // every token under the current execution graph — a statement about
        // the graph, never about the representation's byte width.
        "  dense (always-on, full-read) {:>7.2} GB -> {:>5.1} tok/s",
        l.dense_bytes / 1e9,
        l.tok_s_dense_only
    );
    println!("  routed experts      {:>7.2} GB", l.routed_bytes / 1e9);
    println!(
        "  TOTAL               {:>7.2} GB -> {:>5.1} tok/s",
        l.total_bytes / 1e9,
        l.tok_s_total
    );
    println!();
    // Two different shares, printed together on purpose. The param share is
    // representation-independent; the BYTE share is what a bandwidth lever
    // actually acts on, and they diverge by 28 points on today's K3 because
    // dense is BF16 and routed is MXFP4. Printing only the first next to a
    // table of bytes is how a 53.6% reads as though it described traffic.
    // K3-DENSE-1. The census is the ledger's own answer to "what IS the
    // dense side", and it is ordered by ACTIVATED bytes — `embed_tokens` is
    // the joint-largest RESIDENT family and nearly the smallest activated
    // one, so ordering by footprint would put it at the top of a traffic
    // report.
    println!();
    println!(
        "--- dense families, by activated bytes/token [{}] ---",
        p.dense.label(geom)
    );
    println!(
        "  {:<22} {:>9} {:>7} {:>10}  access",
        "family", "activated", "share", "resident"
    );
    for f in &geom.dense_census.families {
        println!(
            "  {:<22} {:>7.2} GB {:>6.1}% {:>8.2} GB  {}",
            f.name,
            f.activated_bytes() as f64 / 1e9,
            100.0 * geom.dense_census.activated_share(&f.name),
            f.resident_bytes() as f64 / 1e9,
            if f.access.is_full_read() {
                "full read"
            } else {
                "gather"
            }
        );
    }
    println!(
        "  {:<22} {:>7.2} GB {:>6} {:>8.2} GB",
        "TOTAL",
        geom.dense_census.activated_bytes() as f64 / 1e9,
        "100.0%",
        geom.dense_census.resident_bytes() as f64 / 1e9
    );
    println!(
        "  resident exceeds activated by {:.2} GB — the embed table that is \
         never read in full",
        (geom.dense_census.resident_bytes() - geom.dense_census.activated_bytes()) as f64 / 1e9
    );
    println!(
        "  layer topology: {} dense + {} MoE = {} (first_k_dense_replace)",
        geom.topology.n_dense(),
        geom.topology.n_moe(),
        geom.topology.n_layers
    );
    println!();
    println!(
        "  dense is {:.1}% of activated params, and {:.1}% of TODAY'S BYTES",
        100.0 * l.dense_share,
        100.0 * l.dense_bytes / l.total_bytes
    );
    println!(
        "  expert-side levers cap at {:.2}x; {:.2}x needed for {} tok/s",
        l.expert_side_ceiling,
        l.reduction_needed.unwrap_or(f64::NAN),
        a.target_tok_s
    );
    println!(
        "  >> even a perfect expert lever leaves {:.1} tok/s",
        l.tok_s_dense_only
    );
    println!(
        "  at CPU-attainable {:.0} GB/s the same read gives {:.1} tok/s",
        frontier::BW_CPU_GB_S,
        frontier::BW_CPU_GB_S * 1e9 * p.dequant_efficiency / l.total_bytes,
    );
    println!();
    println!("--- slice composition (down-row payloads, up folded) ---");
    for s in &slices {
        println!(
            "  {:>3.0} GB: {:>6.0} experts ({:.1}% of bank) -> {:.2} of {} hit/layer uniform",
            s.slice_bytes / 1e9,
            s.resident_experts,
            100.0 * s.resident_fraction,
            s.uniform_hits_per_layer,
            geom.top_k
        );
    }
    println!();
    println!("  the uniform hit count is a NULL, not a forecast — it assumes no routing skew,");
    println!("  and real routing departs from it in BOTH directions. Measured on the DEC-0");
    println!("  Gemma pool (8-of-128, 6.25% activation): a 6.25% resident set covered 42% of");
    println!("  routing events, not 6.25% — uniform understated coverage ~6.8x. It also");
    println!("  OVERstated per-session support 3.4x at 16 steps. Neither magnitude transfers");
    println!("  to K3 (R2: different activation fraction, and quantile balancing flattens);");
    println!("  the sign of the coverage error does, since uniform understates under any skew.");
    println!("  Measure the real curve with `larql k3-ledger freq-mass --pool <capture>`.");
    Ok(())
}

/// The composed ceiling at the cheapest exact format, grouped experts included.
///
/// Derived rather than quoted: the frontier's own rows sit either side of this
/// bar, and a stale literal here would reintroduce exactly the mislabelling the
/// scenario banner exists to prevent.
fn exact_floor_tok_s(geom: &K3Geometry) -> f64 {
    use super::classes::{self, KernelClass, Scenario};

    let bits = super::serving_format::exact_floor().all_in_bits();
    let rows = classes::apply(
        &classes::census(geom, bits, bits),
        Scenario::LiftClass(KernelClass::RoutedExpert, classes::GROUPED_ROUTED_ETA),
    );
    classes::compose(rows, classes::BW_GB_S, KernelClass::Down.eta()).tok_s
}

pub fn frontier(geom: &K3Geometry, p: &ServingPremises, a: &FrontierArgs, as_json: bool) -> R {
    let p = ServingPremises {
        routed_retention: a.routed_retention.unwrap_or(geom.up_fold_retention()),
        ..*p
    };
    let rows = frontier::frontier(geom, &p, &a.targets);
    let ceiling = frontier::expert_side_ceiling_tok_s(geom, &p);

    let scenario = ScenarioPremises::describe(geom, &p);

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "rows": rows, "expert_side_ceiling_tok_s": ceiling,
                "routed_retention": p.routed_retention,
                // Carried so a row read out of this JSON arrives with its
                // conditions attached rather than as a bare number.
                "scenario": scenario,
                "scenario_banner": scenario::SCENARIO_BANNER,
                "bits_columns_mean": scenario::BITS_COLUMN_MEANING,
                "warnings": scenario.warnings(),
            }))?
        );
        return Ok(());
    }
    header(geom);
    println!();
    println!("=== {} ===", scenario::SCENARIO_BANNER);
    println!(
        "  routed_retention   {:.3}  ({:.2}x credited reduction)",
        scenario.routed_retention, scenario.routed_reduction
    );
    if a.routed_retention.is_none() {
        println!("                     banked up-fold; anything lower is row sparsity, UNMEASURED");
    }
    println!(
        "  routed_bits        {:.2} all-in  (checkpoint MXFP4 bytes, uncompressed)",
        scenario.routed_all_in_bits
    );
    println!(
        "  scalar_efficiency  {:.2}{}",
        scenario.scalar_efficiency,
        if scenario.efficiency_is_ideal() {
            "  <- IDEAL"
        } else {
            ""
        }
    );
    println!(
        "  bandwidth          {:.0} GB/s attainable",
        scenario.bandwidth_gb_s
    );
    println!("  bit convention     rows report BOTH payload and all-in");
    for w in scenario.warnings() {
        println!("  ! {w}");
    }
    println!("  For a whole-model ceiling use `k3-ledger ceilings` (per-class eta).");
    println!(
        "  The exact-format floor is {:.2} tok/s (`k3-ledger formats`) — targets",
        exact_floor_tok_s(geom)
    );
    println!("  above it are fidelity-bounded, not exact-serving, results.");
    println!();
    println!("{:>58}", scenario::BITS_COLUMN_MEANING);
    println!(
        "{:>12} {:>12} {:>13} {:>8} {:>8}  verdict",
        "target tok/s", "budget", "dense budget", "payload", "all-in"
    );
    for r in &rows {
        println!(
            "{:>12.1} {:>11.2}G {:>12.2}G {:>8.2} {:>8.2}  {:?}",
            r.target_tok_s,
            r.budget_bytes / 1e9,
            r.dense_budget_bytes / 1e9,
            r.payload_bits,
            r.all_in_bits,
            r.verdict
        );
    }
    println!(
        "  (bits are per weight; {} excludes block scales, {} includes them)",
        BitConvention::Payload.label(),
        BitConvention::AllIn.label(),
    );
    let undecidable: Vec<_> = rows
        .iter()
        .filter(|r| !frontier::expert_side_can_decide(geom, &p, r.target_tok_s))
        .map(|r| format!("{:.1}", r.target_tok_s))
        .collect();
    if !undecidable.is_empty() {
        println!(
            "  R4: no expert-side experiment can decide {} tok/s",
            undecidable.join(", ")
        );
    }
    if a.zero_out_check {
        println!();
        println!("--- R4 zero-out: routed traffic deleted entirely ---");
        println!(
            "  dense at {:.2} bits [{}] caps decode at {:.1} tok/s",
            p.dense_all_in_bits(geom),
            p.dense.label(geom),
            ceiling
        );
        println!("  >> no expert-side experiment can decide any target above {ceiling:.1}");
    }
    Ok(())
}

pub fn block(geom: &K3Geometry, p: &ServingPremises, a: &BlockArgs, as_json: bool) -> R {
    let p = ServingPremises {
        routed_retention: a.routed_retention.unwrap_or(geom.up_fold_retention()),
        ..*p
    };
    let draft = DraftProfile::new(a.proposal_width, a.mean_accepted);
    let state = match a.state_prefix_ladder_bytes {
        Some(w) => StateTraffic {
            bytes_per_pass: a.state_bytes_per_pass,
            ..StateTraffic::prefix_state_ladder(geom.n_kda_layers, w)
        },
        None => StateTraffic {
            bytes_per_pass: a.state_bytes_per_pass,
            bytes_per_position: a.state_bytes_per_position,
            measured: a.state_bytes_per_pass > 0.0 || a.state_bytes_per_position > 0.0,
        },
    };

    if draft.assumes_perfect() {
        eprintln!(
            "WARNING: proposal width == mean accepted. That is the R5 error \
             (perfect acceptance assumed). Costs key on positions EVALUATED, \
             throughput on positions COMMITTED."
        );
    }

    let rows: Vec<_> = if a.token_loop {
        a.widths
            .iter()
            .map(|&t| {
                let reuse = PhysicalReuse::token_loop(t);
                block::evaluate(geom, &p, &draft, t, reuse, state, a.target_tok_s)
            })
            .collect()
    } else {
        block::sweep(geom, &p, &draft, &a.widths, state, a.target_tok_s)
    };

    let refused: Vec<_> = rows
        .iter()
        .filter(|r| {
            let reuse = PhysicalReuse::assumed_ideal(r.union_equivalents);
            block::dense_alone_refuses(geom, &p, r.accepted, r.width, reuse, state, a.target_tok_s)
        })
        .map(|r| r.width.to_string())
        .collect();

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "rows": rows, "fitted_alpha": draft.fitted_alpha(),
                "provisional": "alpha fitted from ONE observed (width, accepted) pair \
                                under a constant-alpha geometric model; optima are \
                                design points, not performance bars",
            }))?
        );
        return Ok(());
    }
    header(geom);
    println!();
    println!(
        "drafter: {} proposed / {:.2} committed, fitted alpha {:.4}",
        a.proposal_width,
        a.mean_accepted,
        draft.fitted_alpha()
    );
    println!(
        "execution: {}",
        if a.token_loop {
            "TOKEN LOOP (no grouping)"
        } else {
            "ideal grouping ASSUMED (R6, unmeasured)"
        }
    );
    println!(
        "state traffic: {}",
        match (state.bytes_per_pass, state.bytes_per_position) {
            (0.0, 0.0) => "OMITTED — no state term in this budget (owner M1-B)".to_string(),
            (pass, pos) => format!(
                "{:.2} GB/pass + {:.2} GB/position [{}]",
                pass / 1e9,
                pos / 1e9,
                if state.measured {
                    "measured"
                } else {
                    "DERIVED, not measured"
                }
            ),
        }
    );
    println!();
    println!(
        "{:>3} {:>8} {:>8} {:>10} {:>9} {:>9}",
        "T", "A(T)", "u(T)", "GB/token", "tok/s", "G_r need"
    );
    for r in &rows {
        let need = r
            .routed_reduction_needed
            .map(|v| format!("{v:>9.2}"))
            .unwrap_or_else(|| "      inf".into());
        println!(
            "{:>3} {:>8.2} {:>8.2} {:>10.2} {:>9.2} {}",
            r.width,
            r.accepted,
            r.union_equivalents,
            r.bytes_per_committed_token / 1e9,
            r.tok_s,
            need
        );
    }
    if let Some(best) = rows
        .iter()
        .filter(|r| r.rho_max.is_some())
        .max_by(|x, y| x.rho_max.partial_cmp(&y.rho_max).unwrap())
    {
        println!();
        println!(
            "  interior optimum at T={} (needs routed reduction {:.2}x)",
            best.width,
            best.routed_reduction_needed.unwrap_or(f64::NAN)
        );
        println!("  >> a drafter's SHIPPED width is not its best width");
    }
    if !refused.is_empty() {
        println!();
        println!(
            "  R4: dense traffic alone refuses width(s) {} regardless of the routed side",
            refused.join(", ")
        );
    }
    println!();
    println!("  PROVISIONAL: alpha fitted from one observed pair under a constant-alpha");
    println!("  geometric model. These optima are design points, not performance bars.");
    if rows.iter().any(|r| r.rests_on_assumptions) {
        println!("  RESTS ON ASSUMPTIONS: physical reuse and/or state traffic unmeasured.");
    }
    Ok(())
}

pub fn transcode_scan(
    repo: &fetch::Repo,
    shard: &str,
    a: &super::args::TranscodeScanArgs,
    as_json: bool,
) -> R {
    use super::transcode;

    let (header, header_len) = repo.shard_header_with_len(shard)?;
    let tensors = fetch::scale_tensors(&header);
    if tensors.is_empty() {
        return Err("no weight_scale tensors in this shard".into());
    }

    println!("scanning e8m0 exponent spread in {shard}");
    println!(
        "  {} scale tensors present; sampling {}",
        tensors.len(),
        a.tensors.min(tensors.len())
    );

    let mut scans = Vec::new();
    for (name, shape, lo, hi) in tensors.iter().take(a.tensors) {
        let groups_per_row = *shape.last().unwrap_or(&1) as usize;
        let bytes = repo.tensor_bytes_at(shard, header_len, *lo, *hi)?;
        let s = transcode::scan_scales(&bytes, groups_per_row);
        if scans.is_empty() {
            println!(
                "  e.g. {name}: shape {shape:?}, {} groups/row, {} superblocks",
                groups_per_row, s.superblocks
            );
        }
        scans.push(s);
    }
    let m = transcode::merge(&scans);

    if as_json {
        println!("{}", serde_json::to_string_pretty(&m)?);
        return Ok(());
    }

    println!();
    println!("--- exponent field ---");
    println!(
        "  scanned {} tensors, {} superblocks, {} scale bytes",
        m.tensors_scanned, m.superblocks, m.groups_scanned
    );
    println!(
        "  e8m0 range {}..{}  (d exponent {}..{})",
        m.e_min, m.e_max, m.d_exponent_min, m.d_exponent_max
    );
    println!("  sentinels: {} zero, {} NaN", m.zero_scales, m.nan_scales);
    println!();
    println!("--- spread per 256-weight superblock ---");
    for (spread, count) in &m.spread_histogram {
        let pct = 100.0 * *count as f64 / m.superblocks.max(1) as f64;
        let flag = if *spread > transcode::MAX_EXPONENT_SPREAD {
            "  <- EXCEEDS int8 sub-scale"
        } else {
            ""
        };
        println!("  spread {spread:>2}: {count:>9} ({pct:>5.1}%){flag}");
    }
    println!(
        "  max spread {} (limit {})",
        m.max_spread,
        transcode::MAX_EXPONENT_SPREAD
    );
    println!();
    println!("--- constraints ---");
    println!(
        "  C1 spread <= {}   : {}  ({} superblocks over)",
        transcode::MAX_EXPONENT_SPREAD,
        if m.c1_spread_ok { "PASS" } else { "FAIL" },
        m.superblocks_over_spread
    );
    println!(
        "  C2 d in f16 range : {}  ({} superblocks over)",
        if m.c2_f16_ok { "PASS" } else { "FAIL" },
        m.superblocks_over_f16_range
    );
    println!();
    println!("VERDICT: {}", m.verdict());
    Ok(())
}

pub fn ceilings(geom: &K3Geometry, a: &super::args::CeilingsArgs, as_json: bool) -> R {
    use super::classes::{self, KernelClass, Scenario};

    let base = classes::census(geom, a.dense_bits, a.routed_bits);
    let composed = classes::compose(base.clone(), classes::BW_GB_S, KernelClass::Down.eta());

    if as_json {
        println!("{}", serde_json::to_string_pretty(&composed)?);
        return Ok(());
    }

    header(geom);
    println!();
    println!(
        "--- per-class census (dense {:.2} bits, routed {:.2} bits) ---",
        a.dense_bits, a.routed_bits
    );
    println!(
        "{:<28} {:>9} {:>9} {:>6} {:>9}",
        "class", "params B", "GB/token", "eta", "ms/token"
    );
    for r in &composed.rows {
        println!(
            "{:<28} {:>9.2} {:>9.2} {:>6.2} {:>9.2}",
            r.name,
            r.params as f64 / 1e9,
            r.bytes() / 1e9,
            r.eta(),
            r.seconds(classes::BW_GB_S) * 1e3
        );
    }
    println!(
        "{:<28} {:>9} {:>9.2} {:>6} {:>9.2}",
        "TOTAL",
        "",
        composed.total_bytes / 1e9,
        "",
        composed.seconds_per_token * 1e3
    );
    println!();
    let (lo, hi) = classes::observed_range(&composed.rows, classes::BW_GB_S);
    println!("  COMPOSED ceiling {:.2} tok/s", composed.tok_s);
    println!(
        "  observed composition range {lo:.2}-{hi:.2}  (min-max of {} PAIRED per-run \
         compositions, not a component-extrema envelope)",
        classes::ACCEPTED_RUNS
    );
    println!(
        "  scalar-eta quote {:.2} tok/s  -> overstates by {:.2}x",
        composed.scalar_tok_s, composed.scalar_overstates_by
    );
    println!();

    println!("  ETA PROVENANCE: post-shape-census, true cold rotation, 3 repeats.");
    println!("  A class whose range exceeds 5% is NOT decision-grade — do not let it");
    println!("  select a bit-width without more repeats (R0).");
    println!("  eta provenance (R0 — none of these may be quoted bare):");
    for c in [
        KernelClass::AttnProjection,
        KernelClass::GateUp,
        KernelClass::Down,
        KernelClass::RoutedExpert,
    ] {
        let m = c.efficiency();
        println!(
            "    {:.3} +/-{:.4} [{:.2}-{:.2} = {:.0}%] n={}  {:<28} {}{}{}",
            m.central,
            m.std_error,
            m.observed_low,
            m.observed_high,
            100.0 * m.spread_fraction(),
            m.repeats,
            c.label(),
            c.provenance(),
            if c.is_provisional() {
                "   [PROVISIONAL]"
            } else {
                ""
            },
            if m.is_decision_grade() {
                ""
            } else {
                "   [NOT DECISION-GRADE]"
            }
        );
    }
    println!();

    if classes::density_only_bound(a.dense_bits, a.routed_bits) {
        println!(
            "  !! DENSITY-ONLY UPPER BOUND — these etas were measured under {:.4} bpw",
            classes::ETA_MEASURED_UNDER_BITS
        );
        println!(
            "     (Q6_K), and this composition reuses them at {:.4}/{:.4}. R7 forbids",
            a.dense_bits, a.routed_bits
        );
        println!("     carrying eta across a container change: the crossover measured MXFP4");
        println!("     arm D at 0.724 against Q6_K's 0.89 on the routed class. Apply that");
        println!("     penalty before quoting this as reachable — it is an upper bound.");
        println!();
    }

    let ungraded = classes::not_decision_grade(&composed.rows);
    println!("--- R4 best case per proposed lever (ceiling if it FULLY succeeds) ---");
    if !ungraded.is_empty() {
        println!("  REFUSED — this ordering selects the programme, and these classes are");
        println!(
            "  too noisy to select anything (need n>={} and relative SE <= {:.0}%):",
            classes::MIN_REPEATS,
            100.0 * classes::MAX_RELATIVE_SE
        );
        for (c, m) in &ungraded {
            println!(
                "    {:<28} {:.3} +/-{:.4} SE ({:.1}% rel) [{:.2}-{:.2}] n={}",
                c.label(),
                m.central,
                m.std_error,
                100.0 * m.relative_std_error(),
                m.observed_low,
                m.observed_high,
                m.repeats
            );
        }
        println!("  Run `diag_eta_repeats` on those shapes and re-bank before ordering.");
        println!();
        println!(
            "  baseline (measured per-class eta, today's formats) {:.2} tok/s",
            composed.tok_s
        );
        return Ok(());
    }
    let mut scen: Vec<(&str, Vec<classes::ClassRow>)> = vec![
        (
            "exp1 exact MXFP4 layout: all classes -> best eta",
            classes::apply(&base, Scenario::AllClassesAtBestEta),
        ),
        (
            "exp3 3-bit KDA attention",
            classes::apply(&base, Scenario::Rebits("KDA attention projections", 3.0)),
        ),
        (
            "exp3+ 2.5-bit KDA attention",
            classes::apply(&base, Scenario::Rebits("KDA attention projections", 2.5)),
        ),
        (
            "K3a grouped experts: routed 0.64 -> 0.89 [MEASURED]",
            classes::apply(
                &base,
                Scenario::LiftClass(KernelClass::RoutedExpert, classes::GROUPED_ROUTED_ETA),
            ),
        ),
        (
            "exp3+exp4 stacked: 2.5-bit KDA + expert shape fixed",
            classes::apply_all(
                &base,
                &[
                    Scenario::Rebits("KDA attention projections", 2.5),
                    Scenario::LiftClass(KernelClass::RoutedExpert, classes::GROUPED_ROUTED_ETA),
                ],
            ),
        ),
        (
            "K3a+ grouped experts to the roofline (0.95)",
            classes::apply(&base, Scenario::LiftClass(KernelClass::RoutedExpert, 0.95)),
        ),
        (
            "R4 floor: routed experts ZEROED",
            classes::apply(&base, Scenario::Zero("routed experts (top-k)")),
        ),
        (
            "STACKED: 2.5-bit KDA + every class at best eta",
            classes::apply_all(
                &base,
                &[
                    Scenario::Rebits("KDA attention projections", 2.5),
                    Scenario::AllClassesAtBestEta,
                ],
            ),
        ),
        (
            "STACKED + routed experts also zeroed (absolute ceiling)",
            classes::apply_all(
                &base,
                &[
                    Scenario::Rebits("KDA attention projections", 2.5),
                    Scenario::AllClassesAtBestEta,
                    Scenario::Zero("routed experts (top-k)"),
                ],
            ),
        ),
    ];
    // Each scenario carries its OWN band. Quoting the baseline's range beside a
    // scenario's central value is the R0 failure in miniature — the band would
    // not even contain the number it sits next to.
    let mut scored: Vec<(String, f64, f64, f64)> = scen
        .drain(..)
        .map(|(n, rows)| {
            let c = classes::compose(rows.clone(), classes::BW_GB_S, KernelClass::Down.eta());
            let (lo, hi) = classes::observed_range(&rows, classes::BW_GB_S);
            (n.to_string(), c.tok_s, lo, hi)
        })
        .collect();
    scored.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    for (name, tok, lo, hi) in &scored {
        let gain = tok / composed.tok_s;
        println!(
            "  {:<48} {:>6.2} [{:.2}-{:.2}]  ({:+.0}%)",
            name,
            tok,
            lo,
            hi,
            100.0 * (gain - 1.0)
        );
    }
    println!();
    println!(
        "  baseline (measured per-class eta, today's formats) {:.2} tok/s",
        composed.tok_s
    );
    println!("  Order the programme by these, not by how interesting each sounds.");
    Ok(())
}

pub fn symbol_census(
    repo: &fetch::Repo,
    shard: &str,
    a: &super::args::SymbolCensusArgs,
    as_json: bool,
) -> R {
    use super::symbol_census::{self as sc, SymbolCounts};

    let (hdr, hdr_len) = repo.shard_header_with_len(shard)?;
    let all = fetch::packed_expert_tensors(&hdr);
    let wanted: Vec<_> = all
        .into_iter()
        .filter(|(name, ..)| {
            a.experts.iter().any(|e| {
                name.split(".experts.")
                    .nth(1)
                    .and_then(|r| r.split('.').next())
                    .is_some_and(|idx| idx == e.to_string())
            })
        })
        .collect();
    if wanted.is_empty() {
        return Err(format!("no packed expert tensors for {:?} in {shard}", a.experts).into());
    }

    let mut global = SymbolCounts::default();
    let mut nibbles: Vec<(usize, Vec<u8>)> = Vec::new();
    for (name, shape, lo, _hi) in &wanted {
        let row_bytes = *shape.get(1).unwrap_or(&0) as usize;
        let take = (a.rows * row_bytes) as u64;
        let packed = repo.tensor_bytes_at(shard, hdr_len, *lo, lo + take)?;
        global.add_packed(&packed);
        let mut n = Vec::with_capacity(packed.len() * 2);
        for b in &packed {
            n.push(b & 0xF);
            n.push(b >> 4);
        }
        nibbles.push((row_bytes * 2, n));
        eprintln!("  read {:.0} KB of {name}", take as f64 / 1e3);
    }

    let tiles: Vec<_> = a
        .blocks
        .iter()
        .map(|&b| {
            let mut merged = sc::TileStats {
                block: b,
                ..Default::default()
            };
            let mut ent_weighted = 0.0;
            for (row_len, n) in &nibbles {
                let t = sc::tile_stats(n, *row_len, b);
                ent_weighted += t.mean_local_entropy * t.tiles as f64;
                merged.tiles += t.tiles;
                merged.fits_two_bit += t.fits_two_bit;
                merged.fits_three_bit += t.fits_three_bit;
                merged.max_cardinality = merged.max_cardinality.max(t.max_cardinality);
                merged.min_cardinality = if merged.min_cardinality == 0 {
                    t.min_cardinality
                } else {
                    merged.min_cardinality.min(t.min_cardinality)
                };
                merged.median_cardinality = merged.median_cardinality.max(t.median_cardinality);
            }
            merged.mean_local_entropy = ent_weighted / merged.tiles.max(1) as f64;
            merged
        })
        .collect();

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "tensors": wanted.iter().map(|(n, ..)| n).collect::<Vec<_>>(),
                "counts": global, "entropy": global.entropy(),
                "entropy_zero_merged": global.entropy_zero_merged(),
                "tiles": tiles,
            }))?
        );
        return Ok(());
    }

    let n = global.total();
    println!();
    println!(
        "=== MXFP4 symbol census: {:.2} M weights, {} tensors ===",
        n as f64 / 1e6,
        wanted.len()
    );
    println!("  distinct symbols     {}", global.distinct());
    println!(
        "  Shannon entropy      {:.4} bits   (fixed-width floor {:.1})",
        global.entropy(),
        sc::FIXED_PAYLOAD_BITS
    );
    println!(
        "  with +/-0 merged     {:.4} bits   <- free, value-exact",
        global.entropy_zero_merged()
    );
    for k in [1usize, 3, 7] {
        println!("  top-{k} coverage       {:.4}", global.top_k_coverage(k));
    }
    println!();
    println!(
        "--- exact variable-rate candidates (payload bits, floor {:.1}) ---",
        sc::FIXED_PAYLOAD_BITS
    );
    let verdict = |b: f64| {
        if sc::beats_fixed_floor(b) {
            "WINS"
        } else {
            "loses"
        }
    };
    let e3 = sc::escape_bpw(3, global.top_k_coverage(7), 256);
    let e2 = sc::escape_bpw(2, global.top_k_coverage(3), 256);
    let rp = sc::residual_plane_bpw(global.optimal_pairing_rare_mass(), 256);
    println!("  3-bit + escape (7 hot)        {e3:.4}  {}", verdict(e3));
    println!("  2-bit + escape (3 hot)        {e2:.4}  {}", verdict(e2));
    println!(
        "  3-bit primary + residual      {rp:.4}  {}   (rare mass {:.3}, optimal pairing)",
        verdict(rp),
        global.optimal_pairing_rare_mass()
    );
    println!(
        "  ideal arithmetic coder        {:.4}  {}   (serial decode, no random access)",
        global.entropy(),
        verdict(global.entropy())
    );
    println!();
    println!("--- per-tile cardinality and the entropy-bias check ---");
    println!(
        "{:>6} {:>9} {:>7} {:>7} {:>7} {:>9} {:>9} {:>8}",
        "block", "tiles", "min", "med", "max", "<=8 syms", "local H", "bias?"
    );
    for t in &tiles {
        let gap = global.entropy() - t.mean_local_entropy;
        println!(
            "{:>6} {:>9} {:>7} {:>7} {:>7} {:>8.4}% {:>9.4} {:>8}",
            t.block,
            t.tiles,
            t.min_cardinality,
            t.median_cardinality,
            t.max_cardinality,
            100.0 * t.fits_three_bit as f64 / t.tiles.max(1) as f64,
            t.mean_local_entropy,
            if sc::local_skew_is_bias_shaped(gap, sc::FP4_CODES, t.block as u64, 0.06) {
                "YES"
            } else {
                "no"
            },
        );
    }
    println!("  'bias? YES' = the local-vs-global gap is explained by plug-in");
    println!("  entropy bias (K-1)/(2B ln2), NOT by real local concentration.");
    println!();

    println!("--- palette mode at each block's BEST-CASE tile ---");
    for t in &tiles {
        match sc::palette_bpw(t.min_cardinality, t.block as u64) {
            Some(b) => println!(
                "  B={:<4} min cardinality {:>2} -> {b:.4} bpw  {}",
                t.block,
                t.min_cardinality,
                if sc::beats_fixed_floor(b) {
                    "WINS"
                } else {
                    "loses even here"
                }
            ),
            None => println!(
                "  B={:<4} min cardinality {:>2} -> mode cannot fire on ANY tile",
                t.block, t.min_cardinality
            ),
        }
    }
    println!();

    println!("--- decode economics: the score is eta/bpw, not bpw ---");
    // Variable-rate candidates change the payload only; they still owe the same
    // compact scale stream the fixed exact floor pays for.
    let floor_fmt = super::serving_format::exact_floor();
    let floor = floor_fmt.all_in_bits();
    let scale_bits = floor_fmt.scale.bits_per_weight();
    for (label, bpw) in [
        ("ideal arithmetic coder", global.entropy() + scale_bits),
        ("3-bit + escape", e3 + scale_bits),
    ] {
        println!(
            "  {label:<24} {bpw:.4} all-in -> needs eta >= {:.3} to beat the {floor:.4} bpw \
             fixed floor at eta {:.2}",
            sc::break_even_eta(bpw, floor, super::classes::GROUPED_ROUTED_ETA),
            super::classes::GROUPED_ROUTED_ETA,
        );
    }
    println!("  A serial-decode format holding eta 0.81 is the bar to clear.");
    Ok(())
}

pub fn kda_graph(
    repo: &fetch::Repo,
    shard: &str,
    a: &super::args::KdaGraphArgs,
    as_json: bool,
) -> R {
    use super::kda_graph::{self as kg, KdaTensor};

    let hdr = repo.shard_header(shard)?;
    let prefix = format!(".layers.{}.self_attn.", a.layer);
    let obj = hdr.as_object().ok_or("shard header is not an object")?;
    let mut tensors: Vec<KdaTensor> = obj
        .iter()
        .filter_map(|(k, v)| {
            let suffix = k.split(&prefix).nth(1)?;
            let shape: Vec<u64> = v
                .get("shape")?
                .as_array()?
                .iter()
                .filter_map(serde_json::Value::as_u64)
                .collect();
            Some(KdaTensor {
                name: suffix.to_string(),
                shape,
                role: kg::role(suffix),
            })
        })
        .collect();
    if tensors.is_empty() {
        return Err(format!(
            "no self_attn tensors for layer {} in {shard} — is it a KDA layer?",
            a.layer
        )
        .into());
    }
    tensors.sort_by(|x, y| x.name.cmp(&y.name));
    let cov = kg::coverage(&tensors);

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "layer": a.layer, "tensors": tensors, "coverage": cov,
                "byte_coverage": cov.byte_coverage(),
            }))?
        );
        return Ok(());
    }

    println!(
        "=== KDA layer {} — shared-input rotation bundle ===",
        a.layer
    );
    println!("traced from the real tensor table, not from the modeling source");
    println!();
    println!("{:<22} {:>18} {:>12}  role", "tensor", "shape", "params");
    for t in &tensors {
        println!(
            "{:<22} {:>18} {:>12}  {:?}",
            t.name,
            format!("{:?}", t.shape),
            t.params(),
            t.role
        );
    }
    println!();
    println!(
        "  bundle covers {} of {} full-rank projections = {:.1}% of projection bytes",
        cov.bundled_full_rank,
        cov.total_full_rank,
        100.0 * cov.byte_coverage()
    );
    println!("  o_proj reads the gated-normed kernel output, so it CANNOT join:");
    println!("  a rotation does not commute with o_norm's per-channel gain.");
    println!();
    println!("  ! MUST also be folded, and easy to miss:");
    for n in &cov.easy_to_miss {
        println!("      {n}");
    }
    println!("    Rotating h without rewriting these silently corrupts the decay");
    println!("    gate and beta — a plausible wrong number, not a crash.");
    println!();
    println!("  ! output-side folding unavailable: q/k/v pass through a depthwise");
    println!("    ShortConvolution + silu, which no per-channel transform commutes with.");
    if let Some(al) = tensors.iter().find(|t| t.name.starts_with("A_log")) {
        use super::kda_a_log::{classify, ALogAxis, ALogVerdict};
        let numel: u64 = al.shape.iter().product();
        let v = classify(numel, kg::NUM_HEADS, kg::HEAD_DIM);
        println!();
        println!(
            "  A_log [{numel}] against num_heads {} / head_dim {}:",
            kg::NUM_HEADS,
            kg::HEAD_DIM
        );
        match v {
            ALogVerdict::Determined(ALogAxis::PerHead) => {
                println!("    per-head — matches fla's documented [HV] contract.");
            }
            ALogVerdict::Determined(ALogAxis::PerChannel) => {
                println!("    ! per-CHANNEL — contradicts fla, whose gate does A_log.view(H,1).");
                println!("      Two readings survive the big tensors and each breaks one small");
                println!("      one; shapes cannot decide. BLOCKER for KDA numerics — resolve");
                println!("      against an oracle using kda_a_log::asymmetric_fixture.");
            }
            ALogVerdict::AmbiguousBySymmetry => {
                println!("    ! ambiguous: num_heads == head_dim, so the length names both.");
            }
            ALogVerdict::Unexplained => {
                println!("    ! length matches neither axis.");
            }
        }
        if v.axis() != Some(ALogAxis::PerHead) {
            println!("      Sibling Kimi-Linear-48B has A_log numel == num_heads; K3 deviates.");
            // Hand over a fixture the two readings provably disagree on, so an
            // oracle run that "passes" cannot do so vacuously.
            let (fh, fk) = (3usize, 4usize);
            let (a, g, dt) = super::kda_a_log::asymmetric_fixture(fh, fk);
            let ph = super::kda_a_log::decay(ALogAxis::PerHead, &a, &g, &dt, fh, fk);
            let pc = super::kda_a_log::decay(ALogAxis::PerChannel, &a, &g, &dt, fh, fk);
            let differing = ph
                .iter()
                .zip(&pc)
                .filter(|(x, y)| (*x - *y).abs() > 1e-12)
                .count();
            println!();
            println!("      Discriminating fixture ({fh}x{fk}, rectangular on purpose):");
            println!(
                "        separates the readings in {differing} of {} cells",
                fh * fk
            );
            println!("        per-head   [0..3] {:.6?}", &ph[..3]);
            println!("        per-channel[0..3] {:.6?}", &pc[..3]);
            println!("        a SQUARE fixture would agree on the diagonal and prove nothing.");
        }
    }
    Ok(())
}

/// Frequency-mass coverage from a local capture pool. Takes no geometry: the
/// instrument is model-agnostic and must keep working on non-K3 traces.
pub fn freqmass(a: &super::args::FreqMassArgs, as_json: bool) -> R {
    use super::freqmass::{self, MassCurveConfig};
    use super::selection_trace::SelectionTrace;
    use crate::commands::primary::dec_bench::capture_format::CapturePool;

    let pool = CapturePool::open(&a.pool)?;
    let trace = SelectionTrace::from_routing_pool(&pool)?;
    let curve = freqmass::mass_curve(
        &trace,
        &MassCurveConfig {
            cache_sizes: a.cache_sizes.clone(),
            lambdas: a.lambdas.clone(),
            fit_fraction: a.fit_fraction,
            seed: a.seed,
        },
    )?;
    let support = freqmass::support_curve(&trace, &a.support_steps);

    // The operating point is a residency fraction, not a count: it is what
    // transfers across banks of different size.
    let op = curve
        .residency_frac
        .iter()
        .enumerate()
        .min_by(|x, y| {
            (x.1 - a.operating_residency)
                .abs()
                .total_cmp(&(y.1 - a.operating_residency).abs())
        })
        .map(|(i, _)| i);

    // Always censused: the cold-tail counts are a measurement DEC-8.4 wants and
    // cost one pass. Only the per-symbol vector is opt-in, since at K3's 92x896
    // it would dwarf everything else in the document.
    let census = super::symbol_mass::census(&trace);
    let census = if a.per_symbol {
        census
    } else {
        census.without_rows()
    };

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "pool": a.pool.display().to_string(),
                "model": pool.manifest.model,
                "curve": curve,
                "support": support,
                "operating_index": op,
                "symbol_mass": census,
            }))?
        );
        return Ok(());
    }
    if a.per_symbol {
        eprintln!("note: --per-symbol emits the row vector into JSON only; add --json");
    }

    println!("=== frequency-mass coverage — {} ===", a.pool.display());
    println!("model    {}", pool.manifest.model);
    println!(
        "trace    {} sessions x {} steps x {} routing strata, width {} of {} ({:.3}% activation)",
        trace.sessions(),
        trace.steps(),
        trace.active_strata().len(),
        trace.width(),
        trace.alphabet(),
        trace.activation_fraction() * 100.0,
    );
    println!("         bank size INFERRED from the trace (max observed symbol + 1)");
    println!(
        "         fit {} steps -> eval {} steps, {} (session, stratum) cells scored",
        curve.fit_steps, curve.eval_steps, curve.cells_scored,
    );
    println!("R2       every number below is conditional on that activation fraction");
    println!();

    if !support.is_empty() {
        println!("support growth — measured against the memoryless-uniform null");
        println!(
            "  {:>6} {:>10} {:>8} {:>10} {:>15}",
            "steps", "measured", "%bank", "uniform", "overstatement"
        );
        for p in &support {
            println!(
                "  {:>6} {:>10.1} {:>7.1}% {:>10.1} {:>14.2}x",
                p.steps,
                p.measured,
                p.measured_frac * 100.0,
                p.uniform,
                p.overstatement,
            );
        }
        println!();
    }

    print!("{:>5} {:>7}", "C", "%bank");
    for arm in &curve.shrinkage {
        print!("{:>9}", format!("λ={:.2}", arm.lambda));
    }
    println!("{:>9}{:>9}{:>9}", "oracle", "null", "mass/res");
    let stat = curve.static_arm().map(|s| s.coverage.clone());
    for (i, &c) in curve.cache_sizes.iter().enumerate() {
        let marker = if Some(i) == op { "*" } else { " " };
        print!("{marker}{:>4} {:>6.1}%", c, curve.residency_frac[i] * 100.0);
        for arm in &curve.shrinkage {
            print!("{:>9.4}", arm.coverage[i]);
        }
        let per = stat
            .as_ref()
            .map(|s| curve.mass_per_residency(s)[i])
            .unwrap_or(f64::NAN);
        println!(
            "{:>9.4}{:>9.4}{:>8.2}x",
            curve.oracle[i], curve.null[i], per
        );
    }
    println!();
    let unobs: Vec<usize> = census
        .coverage_by_stratum
        .iter()
        .map(|c| c.unobserved)
        .collect();
    if let (Some(&lo), Some(&hi)) = (unobs.iter().min(), unobs.iter().max()) {
        println!(
            "cold tail — {} of {} symbol-stratum pairs UNOBSERVED in this capture ({:.1}%),",
            census.unobserved_symbols,
            census.routing_strata * census.alphabet,
            100.0 * census.unobserved_symbols as f64
                / (census.routing_strata * census.alphabet) as f64,
        );
        println!(
            "  spread {lo}-{hi} per stratum — tier per stratum, not globally. Unobserved is NOT"
        );
        println!("  zero probability: it bounds the rate, it does not measure it.");
        println!("  This is STATIC structure varying by layer — not the per-layer ADAPTATION");
        println!("  the causal arm above refutes. Both hold; do not collapse them.");
        println!("  R2 WARNING: quantile balancing exists to flatten exactly this, so of");
        println!("  everything here it is the number least likely to survive to K3.");
        println!();
        println!("  domain-mixture caveat, BY STATISTIC — the sign differs, so it cannot be");
        println!("  stated once for the whole capture:");
        println!("    coverage curve      ANTI-CONSERVATIVE — a mixture is the anti-case for");
        println!("                        locality; a domain slice would beat these numbers.");
        println!("    unobserved count    CONSERVATIVE — a domain-selective layer would show a");
        println!("                        WIDE union across unrelated prompts. Narrow anyway");
        println!("                        is stronger evidence, not weaker.");
        println!();
    }
    println!("  λ=1.00 static slice (pooled prior, leave-one-out) · λ=0.00 causal adaptive cache");
    println!("  oracle ranks by the scored events themselves · null destroys session identity");
    println!("  mass/res = coverage per unit residency — the axis a cold-tier lever moves along");
    println!();

    if let Some(i) = op {
        let (realisable, oracle) = curve.miss_ratios(i).unwrap_or((f64::NAN, f64::NAN));
        let (best_lambda, best) = curve.best_realisable(i).unwrap_or((f64::NAN, f64::NAN));
        let stat_cov = stat.as_ref().map(|s| s[i]).unwrap_or(f64::NAN);
        println!(
            "operating point — C={} ({:.1}% of bank)",
            curve.cache_sizes[i],
            curve.residency_frac[i] * 100.0
        );
        println!(
            "  static coverage        {stat_cov:.4}  (miss {:.4})",
            1.0 - stat_cov
        );
        println!(
            "  best realisable        {best:.4}  at λ={best_lambda:.2}  (miss {:.4})",
            1.0 - best
        );
        println!("  REALISABLE PRIZE       {realisable:.2}x on the miss stream — what adaptation actually buys");
        println!("  oracle bound           {oracle:.2}x — with every benefit of the doubt granted");
        println!(
            "  locality present       {:.4}  (oracle - null; the rest of the oracle gap is counting noise)",
            curve.locality_signal(i)
        );
    }
    Ok(())
}

pub fn formats(as_json: bool) -> R {
    use super::serving_format::{self as sf, Container, ScaleEncoding};

    if as_json {
        println!("{}", serde_json::to_string_pretty(&sf::candidates())?);
        return Ok(());
    }

    println!("=== exact serving containers for a K3 MXFP4 expert bank ===");
    println!("no checkpoint read: this is decided by the source alphabet");
    println!();
    println!("  fp4 magnitudes           +/-{:?}", sf::FP4_MAGNITUDES);
    println!("  alphabet (x2)            {:?}", sf::FP4_INTEGER_ALPHABET);
    println!("  distinct codepoints      {}", sf::distinct_codepoints());
    println!(
        "  PAYLOAD FLOOR            {} bits — no exact fixed-width code beats this",
        sf::min_payload_bits()
    );
    println!(
        "  affine levels required   {}  (gcd of gaps = 1, span -12..12)",
        sf::affine_levels_required()
    );
    println!();

    println!(
        "{:<22} {:>7} {:>8} {:>7} {:>12}",
        "container", "levels", "exact", "bpw", "kernel reach"
    );
    for c in [
        Container::Mxfp4,
        Container::Q4K,
        Container::Q5K,
        Container::Q6K,
        Container::Q8_0,
    ] {
        println!(
            "{:<22} {:>7} {:>8} {:>7.4} {:>12}",
            c.label(),
            c.grid_levels().map_or("LUT".into(), |l| l.to_string()),
            if c.holds_fp4_exactly() { "yes" } else { "NO" },
            c.all_in_bits(),
            format!(
                "{:?}{}",
                c.kernel_maturity(),
                if c.kernel_maturity().is_servable() {
                    ""
                } else {
                    "*"
                }
            ),
        );
    }
    println!();
    println!(
        "  Q4_K is {} levels short — no scan or scale trick rescues it.",
        sf::affine_levels_required() - Container::Q4K.grid_levels().unwrap()
    );
    println!("  Q6_K is the cheapest exact container that can SERVE today.");
    println!("  MXFP4 has Metal kernels (K1 standalone, K2 grouped) but is absent");
    println!("  from QuantMatVec dispatch, so kernels != servable. (* = cannot serve)");
    println!("  Q5_K is exact but needs a kernel written AND ships more bytes");
    println!("  than native MXFP4 would — strictly dominated, never build it.");
    println!();

    println!("--- scale stream: the only remaining exact lever ---");
    println!(
        "{:<38} {:>9} {:>10} {:>11}",
        "encoding", "bpw", "max spread", "exceptions"
    );
    for s in [
        ScaleEncoding::RawE8M0,
        ScaleEncoding::SuperblockBaseDelta2,
        ScaleEncoding::GlobalTable2Bit,
    ] {
        println!(
            "{:<38} {:>9.5} {:>10} {:>11}",
            s.label(),
            s.bits_per_weight(),
            s.max_spread().map_or("any".into(), |m| m.to_string()),
            if s.needs_exception_path() {
                "REQUIRED"
            } else {
                "none"
            },
        );
    }
    println!();
    println!(
        "  measured max spread {} over 1,032,192 superblocks; both compact",
        sf::MEASURED_MAX_SPREAD
    );
    println!(
        "  encodings clear it. The global table fits the observed {}-exponent",
        sf::MEASURED_DISTINCT_EXPONENTS
    );
    println!("  range with ZERO headroom, so its exception path is mandatory.");
    println!();

    println!("--- candidate ladder, cheapest first ---");
    for f in sf::candidates() {
        println!(
            "  {:>7.4} bpw  {:<22} {:<38} {}",
            f.all_in_bits(),
            f.container.label(),
            if f.container == Container::Mxfp4 {
                f.scale.label()
            } else {
                "—"
            },
            if f.is_exact_for_k3() {
                "exact"
            } else {
                "NOT EXACT"
            },
        );
    }
    println!();

    let floor = sf::exact_floor();
    let cut = 1.0 - floor.all_in_bits() / Container::Mxfp4.all_in_bits();
    println!(
        "  EXACT FLOOR {:.5} bpw — {:.1}% under MXFP4, {:.3}x under Q6_K.",
        floor.all_in_bits(),
        100.0 * cut,
        Container::Q6K.all_in_bits() / floor.all_in_bits()
    );
    println!("  Everything below this bar requires approximation, not encoding.");
    Ok(())
}

/// `k3-ledger homogeneity` — read every layer's header and witness that
/// one measured layer represents its family.
pub fn homogeneity(
    repo: &super::fetch::Repo,
    geom: &K3Geometry,
    a: &super::args::HomogeneityArgs,
    as_json: bool,
) -> R {
    use super::homogeneity::HomogeneityWitness;

    let n = if a.limit == 0 {
        geom.n_layers
    } else {
        a.limit.min(geom.n_layers)
    };
    eprintln!("reading {n} layer headers (headers only, no weights)");
    let mut profiles = Vec::with_capacity(n);
    for layer in 0..n {
        profiles.push(super::fetch::layer_profile(repo, &a.shard_template, layer)?);
        if (layer + 1) % 10 == 0 {
            eprintln!("  {} / {n}", layer + 1);
        }
    }
    let w = HomogeneityWitness::build(&profiles, &geom.kda_layer_indices, geom.topology);

    if as_json {
        println!("{}", serde_json::to_string_pretty(&w)?);
        return Ok(());
    }
    println!();
    println!("--- family homogeneity, from {n} layer headers ---");
    for f in &w.families {
        println!(
            "  {:<16} {:>3} layers, ref layer {:>2}  {}",
            f.family,
            f.members,
            f.reference,
            if f.is_homogeneous() {
                "WITNESSED HOMOGENEOUS".to_string()
            } else {
                format!("{} DIVERGENCES", f.divergences.len())
            }
        );
        for d in f.divergences.iter().take(5) {
            println!("      {d:?}");
        }
        if let Some(r) = profiles.iter().find(|p| p.index == f.reference) {
            println!(
                "      reference layer total {:.2} MB",
                r.total_bytes() as f64 / 1e6
            );
        }
    }
    println!();
    if w.all_homogeneous() {
        println!(
            "  VERDICT: the census may multiply ONE measured layer by its family \
             count — that is now a WITNESSED FACT, not an assumption."
        );
    } else {
        println!(
            "  VERDICT: {} divergences — the census's per-family multiplication \
             is NOT sound as stated.",
            w.divergence_count()
        );
    }
    Ok(())
}

/// `k3-ledger actions` — K3-ACTIONS-1, the physical action catalogue.
///
/// Physical opportunity only. Nothing here is a behavioural claim, and a
/// `family (CEILING)` row is not something one authority run can decide.
pub fn actions(geom: &K3Geometry, a: &super::args::ActionsArgs, as_json: bool) -> R {
    use super::actions::{catalogue, whole_side_ceiling, ActionScope, DENSE_CODECS};

    let cat = catalogue(&geom.dense_census, geom.dense_stored_bits);
    if as_json {
        println!("{}", serde_json::to_string_pretty(&cat)?);
        return Ok(());
    }
    header(geom);
    println!();
    println!(
        "--- K3-ACTIONS-1: physical opportunity from {} ---",
        geom.dense_stored_label
    );
    println!("  NO behavioural claim. A CEILING row is not an atomic candidate.");
    println!();
    println!(
        "  {:<18} {:<7} {:<16} {:>10} {:>10}  execution role",
        "family", "codec", "scope", "activated", "resident"
    );
    for act in &cat {
        if !a.ceilings && act.scope != ActionScope::Layer {
            continue;
        }
        let gb = act.activated_saving() as f64 / 1e9;
        if gb < a.min_gb && act.scope != ActionScope::Layer {
            continue;
        }
        println!(
            "  {:<18} {:<7} {:<16} {:>7.2} GB {:>7.2} GB  {}",
            act.family,
            act.codec,
            act.scope.label(),
            gb,
            act.resident_saving() as f64 / 1e9,
            act.role.label()
        );
    }
    let candidates = cat.iter().filter(|a| a.scope.is_atomic_candidate()).count();
    println!();
    println!(
        "  {candidates} atomic candidates (one authority run each); the rest are \
         CEILINGS that aggregate members which must each earn their own admission"
    );
    println!();
    println!("  --- whole-dense-side ceilings (every family moves AND earns it) ---");
    for codec in DENSE_CODECS {
        println!(
            "  {:<7} {:>7.2} GB/token removed of {:.2} GB dense",
            codec.name,
            whole_side_ceiling(&geom.dense_census, geom.dense_stored_bits, codec) as f64 / 1e9,
            geom.dense_census.activated_bytes() as f64 / 1e9
        );
    }
    Ok(())
}
