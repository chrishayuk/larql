//! **Versioned acceptance criteria, and the evidence they judge.**
//!
//! `quality_proven` must never be a boolean somebody sets. It has to
//! mean *passed a named gate*, because acceptance criteria change: they
//! get tightened when a representation ships, loosened for a draft
//! model, and compared across models. A record saying "quality: ok" with
//! no gate behind it is unfalsifiable a month later, and worse, is
//! indistinguishable from one that passed a much weaker bar.
//!
//! So the evidence carries the gate it was judged by, and the verdict is
//! **derived, never stored** — [`QualityEvidence`] holds only the
//! criteria and the measurements, so it is not possible to construct a
//! passing verdict over a failing bank.
//!
//! ## Routing evidence is kept apart from logit evidence
//!
//! Both live in one bank, but they answer different questions, and for a
//! MoE representation the difference is the whole diagnosis:
//!
//! - **logits moved, routing stable** — an arithmetic effect. The same
//!   experts ran and their outputs shifted. Usually smooth in the
//!   representation's error, and usually the thing more bits fix.
//! - **routing moved** — a decision changed. A different expert ran, so
//!   the downstream effect is not a small perturbation of the same
//!   function, and more bits *on the experts* may not be the response at
//!   all; the router or its inputs may be what needs protecting.
//!
//! Collapsing them into one "quality" number would make those two look
//! identical and send the precision-map search in the wrong direction.

use serde::{Deserialize, Serialize};

/// Acceptance criteria, identified so a claim can name what it passed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityGate {
    /// e.g. `kimi-logit-v1`. Changing a threshold means a NEW id.
    pub id: String,
    /// Fewest positions the bank must cover for its tail statistics to
    /// mean anything. A p99 over nineteen positions is not a p99.
    pub positions_min: u64,
    pub kl_p99_max: f64,
    /// Raw discrete-change counts. `None` means **this gate does not
    /// judge on counts** — not that it permits any number of them.
    ///
    /// v1 and v2 set all three. v3 sets none, because the counts were
    /// measured and found unable to separate a coin flip from an
    /// overturned decision: at layer 26, six argmax flips gave up a
    /// median of 0.1 % of probability, and 232 top-10 changes moved a
    /// median of 0.33 % of top-10 mass one rank. The counts remain in
    /// the evidence as diagnostics; they stop being authority.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top1_flip_max: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top10_change_max: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_flip_max: Option<u64>,
    /// **Consequence limits.** The most probability any single argmax
    /// flip may give up, the top-k mass any position may displace at
    /// p99, and how much of the routed mixture a layer may move.
    ///
    /// These are what make v3 stricter where it matters: one large
    /// mixture replacement fails instantly, while a thousand
    /// microscopic eighth-versus-ninth expert swaps pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top1_mass_displaced_max: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top10_mass_displaced_p99_max: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_mixture_mass_p99_max: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_mixture_mass_max: Option<f64>,
    /// Least baseline mass the bank's truncation must have covered at
    /// its WORST position, for the KL above to be judged at all.
    ///
    /// `None` — v1's shape — means the gate does not ask, and a bank
    /// with no coverage record is judged on its other criteria.
    /// Introducing this on an existing gate id would silently re-date
    /// every claim that ever cited it, so it arrives with a new id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub covered_mass_min: Option<f64>,
}

/// What the bank measured about the output distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogitEvidence {
    pub kl_p50: f64,
    pub kl_p95: f64,
    pub kl_p99: f64,
    pub max_logit_delta: f64,
    /// Positions where the argmax changed.
    pub top1_flips: u64,
    /// Positions where the top-10 SET or its order changed.
    pub top10_changes: u64,
}

/// What the bank measured about routing — deliberately separate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingEvidence {
    /// Selected-expert-id changes, summed over positions and layers.
    pub route_flips: u64,
    /// Positions where ANY layer routed differently. A handful of flips
    /// concentrated in one position is a different fact from the same
    /// count spread across all of them.
    pub positions_with_route_change: u64,
    /// Layers that ever routed differently, so a fix can be scoped by
    /// depth instead of applied to the whole role.
    pub layers_with_route_change: u64,
    /// **How CLOSE the routing decisions that changed actually were.**
    ///
    /// The selection-score gap, in the BASELINE arm, between the last
    /// selected expert and the best unselected one, at each layer whose
    /// route changed — so a small value means the perturbation crossed
    /// a near-tie and a large one means it overturned a decision the
    /// router was confident about.
    ///
    /// Counting route changes cannot tell those apart, and they are not
    /// the same behavioural event. `None` when nothing changed, or when
    /// the bank was built without score evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_margin: Option<Distribution>,
    /// **How much MIXTURE MASS the changed routes moved**, as a
    /// fraction of the layer's routed combine weight.
    ///
    /// A swap of the 8th expert for the 9th at nearly equal weight
    /// moves almost nothing; replacing a heavily-weighted expert moves
    /// a lot. The count is identical in both cases, which is why the
    /// mass is carried separately.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_weight_mass_moved: Option<Distribution>,
    /// The SHALLOWEST layer that ever routed differently, if any.
    ///
    /// The diagnostic that separates two failure modes a count cannot:
    /// a perturbed layer changing its OWN routing (the candidate's
    /// experts were selected differently) versus a perturbed layer
    /// leaving its own routing intact and moving a LATER router's input
    /// — a cascade. Measured on Kimi layer 1 at Q6_K, the layer-1
    /// expert union was identical while twenty-five later layers moved,
    /// which is the second and needs a different response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_layer_with_route_change: Option<u64>,
}

/// A measured quantity's shape, not just its extremes.
///
/// Carried whole because "the worst case was large" and "most cases
/// were large" call for different responses, and a single number
/// cannot distinguish them — the question these exist to answer is
/// whether a criterion is failing on a few real events or on a mass of
/// marginal ones.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Distribution {
    pub count: u64,
    pub min: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

impl Distribution {
    /// Nearest-rank percentiles over `values`, which this sorts.
    ///
    /// Nearest-rank for the same reason the KL percentiles are: a
    /// reported value should be one some observation actually produced.
    pub fn of(values: &mut [f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        values.sort_by(f64::total_cmp);
        let at = |p: f64| {
            let rank = (p * values.len() as f64).ceil().max(1.0) as usize;
            values[rank.min(values.len()) - 1]
        };
        Some(Self {
            count: values.len() as u64,
            min: values[0],
            p50: at(0.50),
            p95: at(0.95),
            p99: at(0.99),
            max: values[values.len() - 1],
        })
    }
}

/// One bank of measurements over a fixed token sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityBank {
    pub positions: u64,
    pub logits: LogitEvidence,
    pub routing: RoutingEvidence,
    /// Smallest baseline probability mass any position's truncation
    /// covered — how much of the distribution the KL above can SEE.
    ///
    /// Recorded because a KL over a truncation that covered a third of
    /// the mass is a different measurement from one that covered all of
    /// it, and nothing else in the bank distinguishes them. Measured on
    /// the real bank: a teacher-forced sequence's FIRST position has no
    /// context and is near-flat over 163,840 ids, so a short truncation
    /// sees almost none of it — top-128 covered 0.307, top-2048 0.729.
    ///
    /// `None` for a bank whose builder did not record it. Judged only
    /// by gates that ask for it — see [`kimi_logit_v2`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_covered_mass: Option<f64>,
    /// **How close the top-10 orderings that changed actually were.**
    ///
    /// The baseline's rank-10-minus-rank-11 logit gap at each position
    /// whose top-10 changed. The top-10 criterion counts any change
    /// including a reordering, and a reordering of two near-tied
    /// candidates is not the same event as a genuine preference
    /// change — this is the evidence that tells them apart.
    ///
    /// EVIDENCE, not a criterion: no gate reads it yet, deliberately,
    /// because a threshold set before the distribution is known is a
    /// guess wearing a version number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top10_margin: Option<Distribution>,
    /// The CANDIDATE's gap at the same two ids. Read against
    /// [`Self::top10_margin`]: if the boundary was near-tied and the
    /// candidate's gap is comparably small, the ordering flipped
    /// because the two were indistinguishable, not because the model's
    /// preference changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top10_candidate_margin: Option<Distribution>,
    /// Probability mass displaced across the top-k, 0 = none, 1 =
    /// disjoint. The top-k analogue of
    /// [`RoutingEvidence::route_weight_mass_moved`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top10_mass_displaced: Option<Distribution>,
    /// Furthest rank move of any id. A swap of ranks 10 and 11 is 1; a
    /// candidate arriving from rank 400 is 390.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top10_rank_displacement: Option<Distribution>,
    /// **How much margin the argmax flips actually crossed.** The
    /// strictest criterion in the contract, and the last one to get
    /// severity evidence: a coin-flip between near-equal candidates and
    /// an overturned confident choice are both "one flip" without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top1_margin: Option<Distribution>,
    /// The candidate's gap over the same pair.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top1_candidate_margin: Option<Distribution>,
    /// Probability the baseline gave up by switching winner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top1_mass_displaced: Option<Distribution>,
}

/// Which criterion a bank failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Criterion {
    Positions,
    KlP99,
    Top1Flips,
    Top10Changes,
    RouteFlips,
    /// An argmax flip gave up more probability than a near-tie would.
    Top1Displacement,
    /// A top-k reordering displaced more mass than a near-tie would.
    TopKDisplacement,
    /// A routing change moved more of the routed mixture than a
    /// near-boundary swap would.
    RouteDisplacement,
    /// The bank's KL was blind to too much of the distribution, or did
    /// not record how much it saw. A gate that asks for coverage and
    /// gets `None` must fail rather than assume the truncation was
    /// wide enough.
    CoveredMass,
}

/// The exact quantity a [`super::constraint::Margin`] reports — the JOIN
/// KEY between a constraint vector, the calibration registry, and a
/// candidate's proxy observations.
///
/// **Finer than [`Criterion`] on purpose.** `RouteDisplacement` carries
/// both a p99 and a max limit and they bind independently: at 256
/// positions the p99 is a maximum wearing a percentile's name while the
/// max is exactly what it says, so keying evidence on the criterion
/// would hand the thin percentile the max's confidence.
///
/// **An enum and not a string, on purpose.** BS2-F2: the registry keyed
/// `"route flip rate"` while the vector emitted `"route flips"`. The
/// lookup missed, `SearchCalibrationRegistry::evidence_for` fell through
/// to its `is_priceable()` arm, and a COUNT — always `Measured` —
/// returned `Direct`. That silently PRICED the one statistic ROUTE-CAL-1
/// calibrated as ordering-only, which is the failure the whole evidence
/// ladder exists to prevent. Two of the three keys matched, so the
/// mechanism looked like it worked. A typed key cannot drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Statistic {
    KlP99,
    Top1Flips,
    Top10Changes,
    RouteFlips,
    Top1MassDisplaced,
    Top10MassDisplacedP99,
    RouteMixtureMassP99,
    RouteMixtureMassMax,
    Positions,
    CoveredMass,
}

impl Statistic {
    /// The human label, for traces. Not a key — nothing joins on this.
    pub fn label(self) -> &'static str {
        match self {
            Self::KlP99 => "kl p99",
            Self::Top1Flips => "top-1 flips",
            Self::Top10Changes => "top-10 changes",
            Self::RouteFlips => "route flips",
            Self::Top1MassDisplaced => "top-1 probability given up",
            Self::Top10MassDisplacedP99 => "top-10 mass displaced at p99",
            Self::RouteMixtureMassP99 => "routed mixture moved at p99",
            Self::RouteMixtureMassMax => "routed mixture moved at max",
            Self::Positions => "positions",
            Self::CoveredMass => "covered mass at the worst position",
        }
    }
}

impl std::fmt::Display for Statistic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl Criterion {
    pub fn name(self) -> &'static str {
        match self {
            Criterion::Positions => "positions",
            Criterion::KlP99 => "kl_p99",
            Criterion::Top1Flips => "top1_flips",
            Criterion::Top10Changes => "top10_changes",
            Criterion::RouteFlips => "route_flips",
            Criterion::Top1Displacement => "top1_mass_displaced",
            Criterion::TopKDisplacement => "top10_mass_displaced",
            Criterion::RouteDisplacement => "route_mixture_mass",
            Criterion::CoveredMass => "min_covered_mass",
        }
    }
}

/// The outcome of judging a bank by a gate.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityVerdict {
    pub gate_id: String,
    /// Empty means passed. Each entry names the criterion and what it
    /// saw against what it required.
    pub failures: Vec<(Criterion, String)>,
}

impl QualityVerdict {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    pub fn describe(&self) -> String {
        if self.passed() {
            return format!("passed {}", self.gate_id);
        }
        let f: Vec<String> = self
            .failures
            .iter()
            .map(|(c, d)| format!("{}: {d}", c.name()))
            .collect();
        format!("failed {} — {}", self.gate_id, f.join("; "))
    }
}

impl QualityGate {
    /// Whether the absence of a displacement distribution means
    /// "nothing changed" rather than "nothing was measured".
    ///
    /// A bank with zero route flips legitimately has no routing
    /// displacement to report, and must not fail for it. A bank that
    /// changed things and recorded no severity is a different matter.
    fn nothing_changed(bank: &QualityBank, criterion: Criterion) -> bool {
        match criterion {
            Criterion::Top1Displacement => bank.logits.top1_flips == 0,
            Criterion::TopKDisplacement => bank.logits.top10_changes == 0,
            Criterion::RouteDisplacement => bank.routing.route_flips == 0,
            _ => false,
        }
    }

    /// Judge a bank. Every criterion is checked, not short-circuited, so
    /// a report says everything that is wrong rather than the first
    /// thing.
    pub fn evaluate(&self, bank: &QualityBank) -> QualityVerdict {
        let mut failures = Vec::new();
        if bank.positions < self.positions_min {
            failures.push((
                Criterion::Positions,
                format!("{} < {} required", bank.positions, self.positions_min),
            ));
        }
        if bank.logits.kl_p99 > self.kl_p99_max {
            failures.push((
                Criterion::KlP99,
                format!("{:.3e} > {:.3e}", bank.logits.kl_p99, self.kl_p99_max),
            ));
        }
        for (limit, got, criterion) in [
            (
                self.top1_flip_max,
                bank.logits.top1_flips,
                Criterion::Top1Flips,
            ),
            (
                self.top10_change_max,
                bank.logits.top10_changes,
                Criterion::Top10Changes,
            ),
            (
                self.route_flip_max,
                bank.routing.route_flips,
                Criterion::RouteFlips,
            ),
        ] {
            if let Some(limit) = limit {
                if got > limit {
                    failures.push((criterion, format!("{got} > {limit}")));
                }
            }
        }
        // Consequence. A criterion the gate asks for and the bank did
        // not record FAILS: an unmeasured consequence is not a small
        // one, and defaulting it to zero is the unfalsifiable claim
        // this module exists to prevent.
        for (limit, observed, criterion, what) in [
            (
                self.top1_mass_displaced_max,
                bank.top1_mass_displaced.map(|d| d.max),
                Criterion::Top1Displacement,
                "top-1 probability given up",
            ),
            (
                self.top10_mass_displaced_p99_max,
                bank.top10_mass_displaced.map(|d| d.p99),
                Criterion::TopKDisplacement,
                "top-10 mass displaced at p99",
            ),
            (
                self.route_mixture_mass_p99_max,
                bank.routing.route_weight_mass_moved.map(|d| d.p99),
                Criterion::RouteDisplacement,
                "routed mixture moved at p99",
            ),
            (
                self.route_mixture_mass_max,
                bank.routing.route_weight_mass_moved.map(|d| d.max),
                Criterion::RouteDisplacement,
                "routed mixture moved at max",
            ),
        ] {
            let Some(limit) = limit else { continue };
            match observed {
                // Nothing changed at all, so nothing was displaced.
                None if Self::nothing_changed(bank, criterion) => {}
                None => failures.push((
                    criterion,
                    format!("{what} was not recorded, and this gate requires <= {limit:.4}"),
                )),
                Some(got) if got <= limit => {}
                Some(got) => failures.push((criterion, format!("{what} {got:.4} > {limit:.4}"))),
            }
        }
        if let Some(required) = self.covered_mass_min {
            match bank.min_covered_mass {
                Some(got) if got >= required => {}
                Some(got) => failures.push((
                    Criterion::CoveredMass,
                    format!("{got:.4} < {required:.4} required"),
                )),
                None => failures.push((
                    Criterion::CoveredMass,
                    format!(
                        "not recorded, and this gate requires at least {required:.4} — a KL                          of unknown coverage is not evidence"
                    ),
                )),
            }
        }
        QualityVerdict {
            gate_id: self.id.clone(),
            failures,
        }
    }
}

/// A bank together with the gate it is judged by.
///
/// The verdict is NOT a field. Storing it would allow a record whose
/// stored verdict and stored numbers disagree, which is exactly the
/// unfalsifiable claim this module exists to prevent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityEvidence {
    pub gate: QualityGate,
    pub bank: QualityBank,
}

impl QualityEvidence {
    pub fn verdict(&self) -> QualityVerdict {
        self.gate.evaluate(&self.bank)
    }

    /// **The promotion report: what decided, and what merely happened.**
    ///
    /// The two are separated on purpose. A reader who sees "211 routing
    /// decisions changed" with no context assumes the worst; a reader
    /// who sees that number under DIAGNOSTICS, beside a measured
    /// consequence under AUTHORITY, can tell the divergence was found,
    /// weighed and judged. Hiding the counts would be worse than
    /// either — they are reported in full, they simply do not decide.
    pub fn report(&self) -> String {
        let verdict = self.verdict();
        let failed = |c: Criterion| verdict.failures.iter().any(|(k, _)| *k == c);
        let mut out = format!("QUALITY_GATE: {}\n\nAUTHORITY:\n", self.gate.id);
        // Each row states the MEASURED authority statistic against its
        // bound — the statistic the criterion is actually judged on,
        // never a neighbouring percentile. A bare PASS invites a later
        // reader to reconstruct the comparison from whatever number is
        // nearest to hand, and a p95 cannot establish a p99 bound.
        let against = |measured: Option<f64>, bound: Option<f64>, dir: &str| match (measured, bound)
        {
            (Some(m), Some(b)) => format!("{m:.3e} vs {dir} {b:.3e}"),
            (None, Some(b)) => format!("no changes vs {dir} {b:.3e}"),
            _ => String::new(),
        };
        let route = self.bank.routing.route_weight_mass_moved.as_ref();
        let route_detail = [
            self.gate
                .route_mixture_mass_p99_max
                .map(|b| format!("p99 {}", against(route.map(|d| d.p99), Some(b), "<="))),
            self.gate
                .route_mixture_mass_max
                .map(|b| format!("max {}", against(route.map(|d| d.max), Some(b), "<="))),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("; ");
        let count_detail = format!(
            "top1 {} top10 {} route {}",
            against(
                Some(self.bank.logits.top1_flips as f64),
                self.gate.top1_flip_max.map(|b| b as f64),
                "<=",
            ),
            against(
                Some(self.bank.logits.top10_changes as f64),
                self.gate.top10_change_max.map(|b| b as f64),
                "<=",
            ),
            against(
                Some(self.bank.routing.route_flips as f64),
                self.gate.route_flip_max.map(|b| b as f64),
                "<=",
            ),
        );
        let asked: [(&str, bool, Criterion, String); 7] = [
            (
                "positions",
                true,
                Criterion::Positions,
                format!("{} vs >= {}", self.bank.positions, self.gate.positions_min),
            ),
            (
                "kl_p99",
                true,
                Criterion::KlP99,
                format!(
                    "{:.3e} vs <= {:.3e}",
                    self.bank.logits.kl_p99, self.gate.kl_p99_max
                ),
            ),
            (
                "covered_mass",
                self.gate.covered_mass_min.is_some(),
                Criterion::CoveredMass,
                against(self.bank.min_covered_mass, self.gate.covered_mass_min, ">="),
            ),
            (
                "top1_mass_displaced",
                self.gate.top1_mass_displaced_max.is_some(),
                Criterion::Top1Displacement,
                format!(
                    "max {}",
                    against(
                        self.bank.top1_mass_displaced.as_ref().map(|d| d.max),
                        self.gate.top1_mass_displaced_max,
                        "<=",
                    )
                ),
            ),
            (
                "top10_mass_displaced",
                self.gate.top10_mass_displaced_p99_max.is_some(),
                Criterion::TopKDisplacement,
                format!(
                    "p99 {}",
                    against(
                        self.bank.top10_mass_displaced.as_ref().map(|d| d.p99),
                        self.gate.top10_mass_displaced_p99_max,
                        "<=",
                    )
                ),
            ),
            (
                "route_mass",
                self.gate.route_mixture_mass_p99_max.is_some()
                    || self.gate.route_mixture_mass_max.is_some(),
                Criterion::RouteDisplacement,
                route_detail,
            ),
            (
                "discrete_counts",
                self.gate.top1_flip_max.is_some(),
                Criterion::Top1Flips,
                count_detail,
            ),
        ];
        for (name, asked_for, criterion, detail) in asked {
            if !asked_for {
                continue;
            }
            out.push_str(&format!(
                "  {name:<22} {}   {detail}\n",
                if failed(criterion) { "FAIL" } else { "PASS" }
            ));
        }
        out.push_str(&format!(
            "\nDIAGNOSTICS (recorded, not authoritative):\n  \
             top1 flips             {}\n  top10 changes          {}\n  \
             route flips            {}\n  positions              {}\n",
            self.bank.logits.top1_flips,
            self.bank.logits.top10_changes,
            self.bank.routing.route_flips,
            self.bank.positions,
        ));
        if let Some(l) = self.bank.routing.first_layer_with_route_change {
            out.push_str(&format!("  first changed layer    {l}\n"));
        }
        out
    }

    /// The gate id this evidence passed, if it passed.
    pub fn proven_by(&self) -> Option<&str> {
        self.verdict().passed().then_some(self.gate.id.as_str())
    }

    /// Whether the logits moved while routing stayed put — the two
    /// mechanisms a MoE precision decision has to tell apart.
    pub fn is_arithmetic_only(&self) -> bool {
        self.bank.routing.route_flips == 0
    }
}

/// **`kimi-logit-v1` — the first acceptance contract for Kimi Linear.**
///
/// Named, and therefore FROZEN. If these thresholds turn out too strict
/// or too lax, the answer is `kimi-logit-v2`; editing this function
/// would silently re-date every claim that ever cited v1, which is the
/// one thing a versioned gate exists to prevent.
///
/// The values are a stated PRIOR, not a calibration — no candidate has
/// been through a bank yet, and guessing thresholds from a distribution
/// nobody has seen is how a bad contract gets frozen. They are written
/// as fractions of an 8192-position bank so the reasoning is inspectable:
///
/// | criterion | value | fraction | why |
/// |---|---|---|---|
/// | `positions_min` | 4096 | half the bank | a p99 needs a tail; 19 positions has none |
/// | `kl_p99_max` | 1e-3 nats | — | at the 99th percentile, well under where sampled text visibly diverges |
/// | `top1_flip_max` | 8 | 0.1 % | greedy decoding changes at all here, so it is the strictest of the four |
/// | `top10_change_max` | 82 | 1 % | reordering inside the top-10 is real but rarely decisive |
/// | `route_flip_max` | 82 | 1 % | a routing change is a decision change, held to the same 1 % |
///
/// The bar this must clear before it is used on anything: a NULL arm
/// (BF16 against itself) has to pass it with every count at zero. A gate
/// its own reference cannot satisfy is measuring the harness.
pub fn kimi_logit_v1() -> QualityGate {
    QualityGate {
        id: "kimi-logit-v1".into(),
        positions_min: 4096,
        kl_p99_max: 1e-3,
        top1_flip_max: Some(8),
        top10_change_max: Some(82),
        route_flip_max: Some(82),
        // v1 does not ask about coverage. See `kimi_logit_v2`.
        covered_mass_min: None,
        // Nor about consequence. See `kimi_logit_v3`.
        top1_mass_displaced_max: None,
        top10_mass_displaced_p99_max: None,
        route_mixture_mass_p99_max: None,
        route_mixture_mass_max: None,
    }
}

/// **`kimi-logit-v2` — v1 plus a bank-validity criterion.**
///
/// Same five thresholds, unchanged and deliberately so: this is not a
/// re-tuning, and a candidate's numbers mean exactly what they meant
/// under v1. What changes is that the KL must be a KL *of something* —
/// the bank has to say how much of the baseline distribution its
/// truncation covered, and that has to be most of it.
///
/// A new id rather than a field added to v1, because a criterion
/// changes what "passed this gate" MEANS. Every claim that ever cited
/// v1 was judged without a coverage requirement, and editing v1 in
/// place would silently re-date all of them as though they had met one.
///
/// `covered_mass_min` is **0.60**, and it is a floor on the WORST
/// position rather than the mean. Its justification is the measurement
/// that motivated it: on Kimi's teacher-forced bank the minimum is
/// driven by each sequence's FIRST position, which has no context and
/// is near-flat over 163,840 ids — top-128 covered 0.307 there and
/// top-2048 covered 0.729, while the p99 barely moved (8.425e-2 →
/// 7.984e-2). So 0.60 rejects the truncation that was demonstrably too
/// narrow, admits the one that was demonstrably wide enough, and does
/// not pretend a context-free position can be made sharp.
pub fn kimi_logit_v2() -> QualityGate {
    QualityGate {
        id: "kimi-logit-v2".into(),
        covered_mass_min: Some(0.60),
        ..kimi_logit_v1()
    }
}

/// **`kimi-logit-v3` — judged by CONSEQUENCE, not by discrete-change
/// counts.**
///
/// v1 and v2 asked whether a discrete boundary was crossed. Measurement
/// showed that question cannot separate the two events it most needs
/// to, at every level of the contract:
///
/// | count | what it actually was, measured at layer 26 |
/// |---|---|
/// | 6 argmax flips | median 0.1 % of probability given up, max 0.49 % |
/// | 232 top-10 changes | median 0.33 % of top-10 mass, one rank |
/// | 0 route flips | — |
///
/// while at layer 1 a single routing change could replace 36 % of the
/// routed mixture. Counting scores those identically.
///
/// So v3 keeps the counts as DIAGNOSTICS and drops them as authority,
/// and adds limits on how much actually moved. That is not a loosening:
/// it fails instantly on one large mixture replacement that v1 would
/// have accepted inside its 82-flip allowance, while no longer failing
/// on a thousand microscopic eighth-versus-ninth expert swaps.
///
/// **The thresholds are read off the measured distributions**, which is
/// the whole point of not writing this gate earlier:
///
/// | criterion | value | why that number |
/// |---|---|---|
/// | `kl_p99_max` | 1e-3 | unchanged from v1; layers 25-26 sit under it, 24 just over |
/// | `covered_mass_min` | 0.60 | v2's, unchanged |
/// | `top1_mass_displaced_max` | 0.05 | every late-band flip gave up <= 0.026; a flip surrendering a twentieth of the model's probability is a preference change, not a tie |
/// | `top10_mass_displaced_p99_max` | 0.10 | observed p99 0.022-0.042 across the band; max seen 0.105 |
/// | `route_mixture_mass_p99_max` | 0.15 | observed p99 0.094-0.105 in the late band |
/// | `route_mixture_mass_max` | 0.25 | late band caps at 0.163; layer 1 reaches 0.361 |
///
/// A criterion this gate asks for and the bank did not record FAILS,
/// unless nothing changed at all — an unmeasured consequence is not a
/// small one.
pub fn kimi_logit_v3() -> QualityGate {
    QualityGate {
        id: "kimi-logit-v3".into(),
        top1_flip_max: None,
        top10_change_max: None,
        route_flip_max: None,
        top1_mass_displaced_max: Some(0.05),
        top10_mass_displaced_p99_max: Some(0.10),
        route_mixture_mass_p99_max: Some(0.15),
        route_mixture_mass_max: Some(0.25),
        ..kimi_logit_v2()
    }
}

/// **`kimi-logit-balanced-v1` — noticeable but bounded movement,
/// calibrated from a measured consequence ladder on TWO banks.**
///
/// Not a relaxation ratio over v3: every limit is drawn from the
/// empirical gap between the last candidate whose changes stayed
/// small and local and the first whose consequences changed character,
/// measured at 8,192 positions on the selection bank AND the held-out
/// bank (zero window overlap, never used to choose topology):
///
/// | anchor @ 8192 | kl p99 (sel/held) | worst top-1 give-up | verdict here |
/// |---|---|---|---|
/// | strict map (experts 24-26 + KDA 24,25, all Q8_0) | 5.99e-4 / — | 0.020 | PASS |
/// | wide map (+ KDA 21,22) | 9.65e-4 / 1.23e-3 | 0.055 / 0.028 | PASS |
/// | flagship (experts 20-26 + KDA 20-25 plateau) | 2.38e-3 / 2.60e-3 | 0.094 / 0.058 | PASS |
/// | B3 (experts 16-26) | 4.74e-3 / — | 0.181, 63 severe overturns | **FAIL** |
///
/// The ladder's ORDERING reproduced on the held-out bank with
/// magnitudes shifting ±30-50%, so each limit carries bank-to-bank
/// margin above the flagship's worst bank rather than sitting on one
/// measurement:
///
/// | criterion | limit | why |
/// |---|---|---|
/// | `kl_p99_max` | 3.5e-3 | flagship worst bank 2.60e-3 × drift margin; refuses B3's 4.74e-3 |
/// | `top1_mass_displaced_max` | 0.12 | flagship worst bank 0.094 × margin; refuses B3's 0.181 — the character change IS this dimension |
/// | `top10_mass_displaced_p99_max` | 0.12 | flagship 0.076 both banks + margin |
/// | route limits | unchanged from v3 | measured NON-discriminating in the corridor (p99 0.126-0.134 from strict to B3) — no evidence justifies loosening |
/// | `covered_mass_min` | 0.55 | the held-out bank's flattest position covers 0.577 at top-2048 — a property of that bank, not of any candidate; 0.60 would make the held-out bank unusable while 0.55 still refuses a blind instrument |
///
/// KL is bounded but is deliberately NOT the definition — B3 taught
/// that a diagnostic-benign map can hide dozens of confident overturns
/// that only authority scale reveals, and the wide map taught that a
/// single overturn's severity (0.055) can exceed strict's whole budget
/// while everything else stays local. The contract boundary is where
/// high-consequence top-1 overturns become materially larger and more
/// frequent, and it was chosen on the held-out bank: the selection
/// bank chose the corridor; the held-out bank chose the contract.
pub fn kimi_logit_balanced_v1() -> QualityGate {
    QualityGate {
        id: "kimi-logit-balanced-v1".into(),
        kl_p99_max: 3.5e-3,
        covered_mass_min: Some(0.55),
        top1_mass_displaced_max: Some(0.12),
        top10_mass_displaced_p99_max: Some(0.12),
        ..kimi_logit_v3()
    }
}

#[cfg(test)]
#[path = "quality_tests.rs"]
mod tests;
