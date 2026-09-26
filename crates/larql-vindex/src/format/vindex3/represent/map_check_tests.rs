use super::*;

const LAYERS: u32 = 12;
const PROJECTIONS: &[&str] = &["q_proj", "k_proj", "v_proj", "o_proj"];

fn map_with(exceptions: Vec<Exception>) -> PrecisionMap {
    PrecisionMap {
        name: "test".into(),
        encoding: "NVFP4".into(),
        roles: vec!["decoder-linear".into()],
        exceptions,
    }
}

fn except(
    projection: Option<&str>,
    layers: Option<(u32, u32)>,
    encoding: Option<&str>,
) -> Exception {
    Exception {
        projection: projection.map(Into::into),
        layers,
        encoding: encoding.map(Into::into),
    }
}

/// Every attention projection of a small decoder, all eligible.
fn decoder() -> Vec<String> {
    (0..LAYERS)
        .flat_map(|l| {
            PROJECTIONS
                .iter()
                .map(move |p| format!("{l}.self_attn.{p}.weight"))
        })
        .collect()
}

fn surface(names: &[String]) -> impl Iterator<Item = (Role, &str)> {
    names.iter().map(|n| (Role::DecoderLinear, n.as_str()))
}

#[test]
fn a_map_with_no_exceptions_passes_any_surface() {
    let names = decoder();
    let cov = map_with(vec![]).check_against(surface(&names)).unwrap();
    assert!(cov.matched.is_empty() && cov.decided.is_empty());
}

#[test]
fn a_partially_overlapping_exception_is_legitimate() {
    // The second rule overlaps the first and still governs every late
    // v_proj — overlap is not the defect, deciding nothing is.
    let names = decoder();
    let cov = map_with(vec![
        except(Some("v_proj"), Some((0, 3)), None),
        except(Some("v_proj"), None, Some("Q8_0")),
    ])
    .check_against(surface(&names))
    .unwrap();
    assert_eq!(cov.matched, vec![4, LAYERS as usize]);
    assert_eq!(cov.decided, vec![4, LAYERS as usize - 4]);
}

#[test]
fn reordering_overlapping_exceptions_can_shadow_the_narrower_one() {
    // The same two rules as above, broad first. First match decides, so
    // the broad rule takes every v_proj and the narrow one is dead — and
    // the refusal names the rule that took its tensors.
    let names = decoder();
    let err = map_with(vec![
        except(Some("v_proj"), None, Some("Q8_0")),
        except(Some("v_proj"), Some((0, 3)), None),
    ])
    .check_against(surface(&names))
    .unwrap_err();
    assert_eq!(
        err.defects,
        vec![MapDefect::Shadowed {
            index: 1,
            selector: "v_proj layers 0-3 -> source".into(),
            shadowed_by: vec![0],
        }]
    );
}

#[test]
fn a_selector_that_matches_nothing_is_refused_by_name() {
    let names = decoder();
    let err = map_with(vec![except(Some("v-proj"), None, None)])
        .check_against(surface(&names))
        .unwrap_err();
    assert_eq!(
        err.defects,
        vec![MapDefect::Unmatched {
            index: 0,
            selector: "v-proj -> source".into()
        }]
    );
}

#[test]
fn a_depth_range_the_model_does_not_have_is_unmatched() {
    // The map alone cannot know this; only the surface can.
    let names = decoder();
    let err = map_with(vec![except(None, Some((40, 47)), None)])
        .check_against(surface(&names))
        .unwrap_err();
    assert!(matches!(
        err.defects[..],
        [MapDefect::Unmatched { index: 0, .. }]
    ));
}

#[test]
fn a_rule_whose_every_match_was_already_decided_is_shadowed() {
    let names = decoder();
    let err = map_with(vec![
        except(Some("v_proj"), Some((0, 5)), None),
        except(Some("v_proj"), Some((6, 11)), None),
        except(Some("v_proj"), Some((2, 8)), Some("Q8_0")),
    ])
    .check_against(surface(&names))
    .unwrap_err();
    assert_eq!(
        err.defects,
        vec![MapDefect::Shadowed {
            index: 2,
            selector: "v_proj layers 2-8 -> Q8_0".into(),
            shadowed_by: vec![0, 1],
        }]
    );
}

#[test]
fn a_catch_all_before_other_rules_is_refused_and_they_are_shadowed() {
    let names = decoder();
    let err = map_with(vec![
        except(None, None, None),
        except(Some("q_proj"), None, Some("Q8_0")),
    ])
    .check_against(surface(&names))
    .unwrap_err();
    assert_eq!(
        err.defects,
        vec![
            MapDefect::CatchAllNotLast {
                index: 0,
                later: vec![1]
            },
            MapDefect::Shadowed {
                index: 1,
                selector: "q_proj -> Q8_0".into(),
                shadowed_by: vec![0],
            },
        ]
    );
}

#[test]
fn a_catch_all_is_refused_before_later_rules_even_on_an_empty_surface() {
    let err = map_with(vec![
        except(None, None, None),
        except(Some("q_proj"), None, None),
    ])
    .check_against(std::iter::empty())
    .unwrap_err();
    assert!(err.defects.contains(&MapDefect::CatchAllNotLast {
        index: 0,
        later: vec![1]
    }));
}

#[test]
fn a_trailing_catch_all_is_legitimate() {
    // "compile nothing but q_proj" — the documented use.
    let names = decoder();
    let cov = map_with(vec![
        except(Some("q_proj"), None, Some("NVFP4")),
        except(None, None, None),
    ])
    .check_against(surface(&names))
    .unwrap();
    assert_eq!(
        cov.decided,
        vec![LAYERS as usize, names.len() - LAYERS as usize]
    );
}

#[test]
fn tensors_of_a_role_the_map_does_not_compile_decide_nothing() {
    // resolve() answers source for these before consulting an exception,
    // so an exception that only they match is dead.
    let names = ["0.mlp.gate.weight".to_string()];
    let err = map_with(vec![except(Some("gate"), None, None)])
        .check_against(names.iter().map(|n| (Role::Router, n.as_str())))
        .unwrap_err();
    assert!(matches!(err.defects[..], [MapDefect::Unmatched { .. }]));
}

#[test]
fn every_defect_is_reported_not_only_the_first() {
    let names = decoder();
    let err = map_with(vec![
        except(Some("v-proj"), None, None),
        except(Some("k_proj"), None, None),
        except(Some("k_proj"), Some((0, 1)), Some("Q8_0")),
    ])
    .check_against(surface(&names))
    .unwrap_err();
    assert_eq!(err.defects.len(), 2);
    let text = err.to_string();
    assert!(text.starts_with("precision map `test` refused:"), "{text}");
    assert!(
        text.contains("#0 `v-proj -> source` matches no eligible tensor"),
        "{text}"
    );
    assert!(
        text.contains("#2 `k_proj layers 0-1 -> Q8_0` decides nothing"),
        "{text}"
    );
}

#[test]
fn the_check_agrees_with_resolve_on_who_decides() {
    // The replay must be the same first-match program resolve runs, or
    // the check would certify a map against a different semantics.
    let names = decoder();
    let m = map_with(vec![
        except(Some("v_proj"), Some((0, 3)), None),
        except(Some("v_proj"), None, Some("Q8_0")),
        except(None, Some((10, 11)), Some("Q6_K")),
    ]);
    let cov = m.check_against(surface(&names)).unwrap();
    let decided_by_q8 = names
        .iter()
        .filter(|n| {
            m.resolve(Role::DecoderLinear, n) == super::super::map::Precision::Compiled("Q8_0")
        })
        .count();
    assert_eq!(cov.decided[1], decided_by_q8);
}
