//! `larql k3-ledger` — checkpoint-derived serving arithmetic (docs/dec-funnel.md).
//!
//! Answers what a K3-class model costs to serve, from the checkpoint's own
//! tensor table rather than from estimates. Every rung is zero-compute: two HTTP
//! range requests against a 1.5 TB repository, then division.
//!
//! Module map:
//!
//!   access    — HOW a tensor is touched: FullRead / Gather / Routed, plus
//!               the dense/MoE layer topology. Physical existence is not
//!               execution touch (pure, gated).
//!   actions   — K3-ACTIONS-1: the weight-free physical action catalogue.
//!               Representation is not execution role (pure, gated).
//!   args      — clap surface (`budget` / `touch` / `frontier` / `block`).
//!   geometry  — measured checkpoint shape; nothing derived (pure, gated).
//!   budget    — DEC-8.0 miss budget and read-granularity (pure, gated).
//!   touch     — DEC-8.6 resident weight-touch ledger (pure, gated).
//!   frontier  — DEC-8.7a dense-precision frontier, R4 ceiling (pure, gated).
//!   block     — DEC-9.2 speculative block economics, R5/R6 (pure, gated).
//!   selection_trace — generic "which symbols did this position pick" (pure).
//!   symbol_mass — the measured per-symbol routing mass vector, counts kept
//!               alongside probabilities; DEC-8.4's allocator input (pure).
//!   freqmass  — frequency-mass coverage: static / adaptive / oracle / null.
//!               Model-agnostic and symbol-agnostic; grades the resident bank,
//!               the static slice, and compact-dense with one instrument (pure).
//!   homogeneity — is ONE measured layer representative of its family?
//!               Witnesses it from headers before any weight is read (pure).
//!   kda_a_log — is A_log per-head or per-channel? Fails closed (pure).
//!   kda_graph — which KDA projections share one input, for rotation (pure).
//!   scenario  — the premises a `frontier` row is conditional on (pure).
//!   serving_format — which containers hold MXFP4 exactly, by counting (pure).
//!   symbol_census  — is there an exact VARIABLE-rate code below 4 bits (pure).
//!   cond_census    — ETC-0A statistics: is an expert cheaper GIVEN another
//!                  expert? Conditional entropy on untouched symbols (pure).
//!   cond_arms — ETC-0A arm design: prototype, parents, and the nulls (pure).
//!   column_profile — the one conditioning context that survives the expert
//!                  row permutation; read its header before trusting any
//!                  coordinate-aligned negative result (pure).
//!   transport_bpw  — what a conditional bit-rate is worth on each arm, with
//!                  R4 applied before the codec exists (pure).
//!   scale_stream   — fold the e8m0 scale channel into the census alphabet, so
//!                  the ledger prices payload AND metadata together (pure).
//!   retention — DEC-9A retention oracle gate: Belady/MIN vs LRU vs a best
//!                  fixed set, isolating the temporal prize (pure).
//!   rng       — reproducible PRNG shared by every null in the ladder (pure).
//!   fetch     — HTTP range-GET geometry loading (I/O, excluded).
//!   report    — human-readable rendering (I/O-adjacent, excluded).
//!
//! The pure/runtime split is load-bearing here. Every error this ladder caught
//! was arithmetic over correct measurements, so the arithmetic carries unit
//! tests and the network does not.

pub mod access;
pub mod actions;
pub mod args;
pub mod block;
pub mod budget;
pub mod classes;
pub mod column_profile;
pub mod cond_arms;
pub mod cond_census;
mod cond_report;
pub mod fetch;
pub mod freqmass;
#[cfg(test)]
mod freqmass_tests;
pub mod frontier;
pub mod geometry;
pub mod homogeneity;
pub mod kda_a_log;
pub mod kda_graph;
mod report;
pub mod retention;
mod retention_report;
#[cfg(test)]
mod retention_tests;
pub mod rng;
pub mod scale_stream;
pub mod scenario;
pub mod selection_trace;
pub mod serving_format;
pub mod symbol_census;
pub mod symbol_mass;
pub mod touch;
pub mod transcode;
pub mod transport_bpw;

pub use args::K3LedgerArgs;

pub fn run(args: K3LedgerArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Decided by counting the source alphabet, not by reading the checkpoint —
    // so it must not pay for geometry it never consults.
    if matches!(args.cmd, args::K3LedgerCmd::Formats) {
        return report::formats(args.json);
    }
    // Reads a local capture pool and knows nothing about K3 — must not fetch
    // geometry, or the instrument stops working on every non-K3 trace.
    if let args::K3LedgerCmd::FreqMass(a) = &args.cmd {
        return report::freqmass(a, args.json);
    }
    // Same: a local routing trace, no K3 geometry involved.
    if let args::K3LedgerCmd::Retention(a) = &args.cmd {
        return retention_report::run(a, args.json);
    }

    let repo = fetch::Repo::new(&args.repo)?;
    eprintln!(
        "reading geometry from {} (headers only, no weights)",
        args.repo
    );
    let geom = fetch::load_geometry(&repo, &args.kda_shard, &args.mla_shard)?;

    let premises = frontier::ServingPremises {
        bandwidth_gb_s: args.bandwidth_gb_s,
        dequant_efficiency: args.dequant_efficiency,
        dense: match args.hypothetical_dense_bits {
            Some(all_in_bits) => geometry::DensePricing::Hypothetical { all_in_bits },
            None => geometry::DensePricing::AsStored,
        },
        ..Default::default()
    };

    match &args.cmd {
        args::K3LedgerCmd::Budget(a) => report::budget(&geom, a, args.json),
        args::K3LedgerCmd::Homogeneity(a) => report::homogeneity(&repo, &geom, a, args.json),
        args::K3LedgerCmd::Actions(a) => report::actions(&geom, a, args.json),
        args::K3LedgerCmd::Touch(a) => report::touch(&geom, &premises, a, args.json),
        args::K3LedgerCmd::Frontier(a) => report::frontier(&geom, &premises, a, args.json),
        args::K3LedgerCmd::Block(a) => report::block(&geom, &premises, a, args.json),
        args::K3LedgerCmd::Ceilings(a) => report::ceilings(&geom, a, args.json),
        // Handled above, before geometry is fetched.
        args::K3LedgerCmd::Formats
        | args::K3LedgerCmd::FreqMass(_)
        | args::K3LedgerCmd::Retention(_) => unreachable!(),
        args::K3LedgerCmd::TranscodeScan(a) => {
            report::transcode_scan(&repo, &args.kda_shard, a, args.json)
        }
        args::K3LedgerCmd::SymbolCensus(a) => {
            report::symbol_census(&repo, &args.kda_shard, a, args.json)
        }
        args::K3LedgerCmd::CondCensus(a) => {
            cond_report::run(&repo, &geom, &premises, &args.kda_shard, a, args.json)
        }
        args::K3LedgerCmd::KdaGraph(a) => report::kda_graph(&repo, &args.kda_shard, a, args.json),
    }
}
