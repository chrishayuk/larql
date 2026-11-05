//! REFINE-IDEMPOTENCE — the invariant that fixed V2's compound-drift
//! bug, frozen for the V3 port: the batch refine reads the session's
//! RAW captured residuals, never refined state. Re-refining from the
//! same raws is a fixed point, and later installs rebuild earlier
//! slots from the ORIGINAL captures.

use crate::executor::{Backend, Session};
use crate::parse;
use larql_vindex::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};

/// A word-level tokenizer WITH a whitespace pre-tokenizer, so the
/// canonical prompts of distinct facts tokenize to distinct sequences.
/// (The plain synthetic tokenizer has no pre-tokenizer: every prose
/// prompt collapses to one `[UNK]`, giving every fact an IDENTICAL
/// capture — refine then rightly annihilates them all, which tests the
/// fixture, not the invariant.)
fn word_tokenizer_json() -> String {
    let vocab = r#""[UNK]":0,"The":1,"of":2,"is":3,"a":4,"b":5,"c":6,"[5]":7,"[6]":8"#;
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":{{\"type\":\"Whitespace\"}},\
         \"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{vocab}}},\"unk_token\":\"[UNK]\"}}}}"
    )
}

fn v3_session() -> (tempfile::TempDir, Session) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "refine-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        word_tokenizer_json(),
    )
    .unwrap();
    let mut session = Session::new();
    let use_stmt = format!(
        "USE \"{}\";",
        container.path().display().to_string().replace('\\', "\\\\")
    );
    session.execute(&parse(&use_stmt).unwrap()).unwrap();
    (container, session)
}

fn run(session: &mut Session, stmt: &str) {
    session
        .execute(&parse(stmt).unwrap())
        .unwrap_or_else(|e| panic!("{stmt}: {e}"));
}

fn overlay_gate(session: &Session, layer: usize, feature: usize) -> Vec<f32> {
    let Backend::Vindex3 { overlay, .. } = &session.backend else {
        panic!("V3 session");
    };
    overlay
        .gate_override_at(layer, feature)
        .expect("composed slot has a gate")
        .to_vec()
}

fn composed_slots(session: &Session) -> Vec<(usize, usize)> {
    let mut slots: Vec<(usize, usize)> = session.raw_install_residuals.keys().copied().collect();
    slots.sort_unstable();
    slots
}

/// Re-refining from the same raw captures is a fixed point: the
/// orchestration feeds `refine_gates` the raw snapshot, so a second
/// pass with the same reference norms reproduces the stored gates and
/// ups bit for bit.
///
/// The reference norms come from `layer_median_norms`, the function the
/// install itself calls. They must not be rebuilt from the stored
/// vectors: `norm(unit(r) * g * GATE_SCALE) / GATE_SCALE` recovers `g`
/// only up to rounding, and whether it lands back on `g` exactly depends
/// on the captured residual's values, so on the BLAS that produced them.
/// That held on OpenBLAS 0.3.26 for slot 0 only (slot 1 was already 2 ULP
/// off) and failed on 0.3.32. The magnitude the stored vectors carry is
/// checked separately, with a derived bound, in
/// `stored_vectors_carry_the_layer_reference_norms`.
#[test]
fn re_refining_from_the_same_raws_is_a_fixed_point() {
    let (_c, mut session) = installed_pair();
    let slots = composed_slots(&session);
    assert_eq!(slots.len(), 2, "two installs at the layer");
    let before: Vec<(Vec<f32>, Vec<f32>)> = slots
        .iter()
        .map(|&(l, f)| (overlay_gate(&session, l, f), overlay_up(&session, l, f)))
        .collect();

    // A second refine over the same session state must change nothing.
    let (g_ref, u_ref) = reference_norms(&session, 1);
    session.refine_layer_from_raw_v3(1, g_ref, u_ref);

    for (i, &(l, f)) in slots.iter().enumerate() {
        assert_eq!(
            overlay_gate(&session, l, f),
            before[i].0,
            "refine from the same raws must be a fixed point at ({l},{f}) (gate)"
        );
        assert_eq!(
            overlay_up(&session, l, f),
            before[i].1,
            "refine from the same raws must be a fixed point at ({l},{f}) (up)"
        );
    }
}

/// Every refined slot stores `unit(refined) * g_ref * GATE_SCALE` as its
/// gate and `unit(refined) * u_ref` as its up, so the stored norms equal
/// the layer's reference norms up to f32 rounding, never exactly.
///
/// Bound, with u = 2^-24 and n = hidden size. Each of the two norm
/// evaluations (the one inside `unit_vector`, and the one here) is a sum of
/// n squares (relative error at most gamma_n), halved by the square root,
/// plus one rounding for the root itself: at most gamma_(n+1). The
/// element-wise chain adds at most four more roundings (divide by the
/// norm, times g_ref, times GATE_SCALE, and the divide by GATE_SCALE
/// here). So the relative error is at most gamma_(2n+6), where
/// gamma_k = k*u / (1 - k*u). The up path has fewer roundings, so the same
/// bound covers it.
#[test]
fn stored_vectors_carry_the_layer_reference_norms() {
    let (_c, session) = installed_pair();
    let (g_ref, u_ref) = reference_norms(&session, 1);
    let scale = crate::executor::tuning::GATE_SCALE;
    for (l, f) in composed_slots(&session) {
        let gate = overlay_gate(&session, l, f);
        let up = overlay_up(&session, l, f);
        let bound = gamma(2 * gate.len() + 6);
        let g_err = ((norm(&gate) / scale) as f64 - g_ref as f64).abs() / g_ref as f64;
        let u_err = (norm(&up) as f64 - u_ref as f64).abs() / u_ref as f64;
        assert!(
            g_err <= bound,
            "gate norm at ({l},{f}) is {g_err:e} relative from g_ref; bound {bound:e}"
        );
        assert!(
            u_err <= bound,
            "up norm at ({l},{f}) is {u_err:e} relative from u_ref; bound {bound:e}"
        );
    }
}

/// The standard floating-point error constant gamma_k for f32.
fn gamma(k: usize) -> f64 {
    let u = f64::from(f32::EPSILON) / 2.0;
    let ku = k as f64 * u;
    ku / (1.0 - ku)
}

/// Two composed installs at layer 1, the state both refine tests start from.
fn installed_pair() -> (tempfile::TempDir, Session) {
    let (c, mut session) = v3_session();
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#,
    );
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "b", "[6]") AT LAYER 1 MODE COMPOSE;"#,
    );
    (c, session)
}

/// The (g_ref, u_ref) the install used at `layer`: the same
/// `layer_median_norms` call, so the test reuses production's factors
/// rather than rebuilding them from stored vectors.
fn reference_norms(session: &Session, layer: usize) -> (f32, f32) {
    let Backend::Vindex3 { runtime, .. } = &session.backend else {
        panic!("V3 session")
    };
    let (g_ref, u_ref, _d_ref) = super::layer_median_norms(runtime, layer).unwrap();
    (g_ref, u_ref)
}

fn overlay_up(session: &Session, layer: usize, feature: usize) -> Vec<f32> {
    let Backend::Vindex3 { overlay, .. } = &session.backend else {
        panic!("V3 session")
    };
    overlay
        .up_override_at(layer, feature)
        .expect("composed slot has an up")
        .to_vec()
}

/// Later installs rebuild earlier slots from the ORIGINAL raw
/// captures: after B lands, A's stored gate equals a fresh
/// `refine_gates` run over the raw snapshot — not a re-projection of
/// A's already-refined gate.
///
/// The decoy cache is pre-seeded SMALL: the miniature's hidden size is
/// 12, and the production decoy set (~20 vectors) spans the whole
/// space — Gram-Schmidt then annihilates every gate and the surviving
/// direction is rounding junk, which is a fixture artifact, not the
/// invariant under test. Three suppressors keep the space
/// well-conditioned so directions are meaningful.
#[test]
fn later_installs_refine_from_original_captures() {
    let (_c, mut session) = v3_session();
    let seeded: Vec<larql_vindex::ndarray::Array1<f32>> = (0..2)
        .map(|i| {
            let mut v = vec![0.0f32; 12];
            v[i] = 1.0;
            larql_vindex::ndarray::Array1::from_vec(v)
        })
        .collect();
    session.decoy_residual_cache.insert(1, seeded);
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#,
    );
    let raw_a = session.raw_install_residuals.clone();
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "b", "[6]") AT LAYER 1 MODE COMPOSE;"#,
    );

    // The raw snapshot still holds A's ORIGINAL capture, untouched.
    for (key, original) in &raw_a {
        assert_eq!(
            session.raw_install_residuals.get(key).unwrap(),
            original,
            "raw captures are immutable inputs"
        );
    }

    // A's stored gate direction is the pure recomputation from raws +
    // decoys — proving refine consumed raw captures.
    let slots = composed_slots(&session);
    let inputs: Vec<larql_vindex::RefineInput> = slots
        .iter()
        .map(|&(l, f)| larql_vindex::RefineInput {
            layer: l,
            feature: f,
            gate: session.raw_install_residuals[&(l, f)].clone(),
        })
        .collect();
    let decoys = session
        .decoy_residual_cache
        .get(&1)
        .cloned()
        .unwrap_or_default();
    let expected = larql_vindex::refine_gates(&inputs, &decoys);
    for refined in expected.gates {
        let stored = overlay_gate(&session, refined.layer, refined.feature);
        let stored_dir = unit(&stored);
        let expected_dir = unit(refined.gate.as_slice().unwrap());
        let cos: f32 = stored_dir
            .iter()
            .zip(&expected_dir)
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            cos > 0.999_999,
            "stored gate at ({},{}) must be the raw recomputation (cos {cos}, \
             stored norm {}, expected norm {}, retained {})",
            refined.layer,
            refined.feature,
            norm(&stored),
            norm(refined.gate.as_slice().unwrap()),
            refined.retained_norm
        );
    }
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = norm(v).max(1e-12);
    v.iter().map(|x| x / n).collect()
}
