use super::*;

fn site(layer: usize, phase: Option<Phase>, step: Option<u64>) -> ExpertGroupSite {
    ExpertGroupSite {
        layer,
        phase,
        step,
        slots: 4,
    }
}

/// `Every` is the only selector that covers an undeclared step — it is
/// the "I do not address tokens at all" case, so there is nothing to
/// refuse.
#[test]
fn every_covers_declared_and_undeclared_steps() {
    assert!(StepSelector::Every.selects(None));
    assert!(StepSelector::Every.selects(Some(0)));
    assert!(StepSelector::Every.selects(Some(9_999)));
}

/// Every step-addressing selector REFUSES an undeclared step rather than
/// treating it as 0. Treating it as 0 would make "skip on token 0" fire
/// on every boundary the driver loop failed to attribute — which is a
/// silent, uncounted intervention.
#[test]
fn step_addressing_selectors_refuse_an_undeclared_step() {
    assert!(!StepSelector::Exactly(0).selects(None));
    assert!(!StepSelector::EveryNth(1).selects(None));
    assert!(!StepSelector::OneOf(vec![0]).selects(None));
}

#[test]
fn exactly_selects_one_step() {
    let s = StepSelector::Exactly(7);
    assert!(s.selects(Some(7)));
    assert!(!s.selects(Some(6)));
    assert!(!s.selects(Some(8)));
}

#[test]
fn every_nth_selects_the_arithmetic_series() {
    let s = StepSelector::EveryNth(3);
    for step in [0u64, 3, 6, 300] {
        assert!(s.selects(Some(step)), "step {step} should be selected");
    }
    for step in [1u64, 2, 4, 5] {
        assert!(!s.selects(Some(step)), "step {step} should not be selected");
    }
}

/// A zero period selects nothing instead of dividing by zero. Silently
/// selecting everything would be the worse failure: a typo'd period would
/// delete every expert group in the model.
#[test]
fn every_nth_zero_selects_nothing() {
    let s = StepSelector::EveryNth(0);
    assert!(!s.selects(Some(0)));
    assert!(!s.selects(Some(1)));
    assert!(!s.selects(None));
}

#[test]
fn one_of_selects_exactly_its_set() {
    let s = StepSelector::OneOf(vec![2, 5, 11]);
    assert!(s.selects(Some(5)));
    assert!(!s.selects(Some(4)));
}

/// The core behaviour: named layers skip, everything else runs.
#[test]
fn mask_skips_only_the_named_layers() {
    let mask = LayerStepMask::new([20, 22]);
    assert_eq!(
        mask.expert_group(&site(20, Some(Phase::Decode), Some(0))),
        ExecutionStrategy::Skip
    );
    assert_eq!(
        mask.expert_group(&site(22, Some(Phase::Decode), Some(0))),
        ExecutionStrategy::Skip
    );
    assert_eq!(
        mask.expert_group(&site(21, Some(Phase::Decode), Some(0))),
        ExecutionStrategy::Canonical,
        "a layer between two targets must still run"
    );
    assert_eq!(
        mask.expert_group(&site(0, Some(Phase::Decode), Some(0))),
        ExecutionStrategy::Canonical
    );
}

/// A phase-restricted mask must not fire in the other phase, and must not
/// fire when no phase was declared. Assuming decode on an undeclared
/// phase is exactly how prefill positions got counted as decode steps
/// once already.
#[test]
fn phase_restriction_refuses_the_wrong_and_the_undeclared_phase() {
    let mask = LayerStepMask::new([20]).with_phase(Phase::Decode);
    assert_eq!(
        mask.expert_group(&site(20, Some(Phase::Decode), Some(0))),
        ExecutionStrategy::Skip
    );
    assert_eq!(
        mask.expert_group(&site(20, Some(Phase::Prefill), Some(0))),
        ExecutionStrategy::Canonical
    );
    assert_eq!(
        mask.expert_group(&site(20, None, Some(0))),
        ExecutionStrategy::Canonical
    );
}

/// The single-layer, single-token form — what the production gate arms.
/// It is the whole point of `Exactly`: "one known layer, one known token"
/// as an exact statement, not a first-visit approximation.
#[test]
fn one_layer_one_token_fires_exactly_once_in_the_address_space() {
    let mask = LayerStepMask::new([20])
        .with_phase(Phase::Decode)
        .with_steps(StepSelector::Exactly(3));
    let mut skips = 0;
    for step in 0..8u64 {
        for layer in 0..24usize {
            if mask.expert_group(&site(layer, Some(Phase::Decode), Some(step)))
                == ExecutionStrategy::Skip
            {
                skips += 1;
            }
        }
    }
    assert_eq!(skips, 1, "exactly one (layer, step) cell may skip");
}

/// Duplicate layers collapse, so a caller that passes `[20, 20]` does not
/// get a mask that looks like it targets two layers in the report.
#[test]
fn duplicate_layers_collapse() {
    let mask = LayerStepMask::new([20, 20, 20]);
    assert!(mask.name().contains("layers=[20]"));
}

/// The name is derived from the configuration, never supplied, so the
/// report line cannot describe a policy other than the one that ran.
#[test]
fn name_describes_the_actual_configuration() {
    let mask = LayerStepMask::new([22, 20])
        .with_phase(Phase::Decode)
        .with_steps(StepSelector::EveryNth(4));
    let name = mask.name();
    assert!(name.contains("layers=[20,22]"), "sorted layers: {name}");
    assert!(name.contains("phase=decode"), "{name}");
    assert!(name.contains("steps=every-4th"), "{name}");
}

/// An unrestricted mask says so in its name, rather than leaving a reader
/// to assume decode-only.
#[test]
fn unrestricted_mask_names_itself_as_any_phase_every_step() {
    let name = LayerStepMask::new([1]).name().to_string();
    assert!(name.contains("phase=any"), "{name}");
    assert!(name.contains("steps=every"), "{name}");
}
