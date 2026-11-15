//! Every site writes the array shape its page states.

use crate::support::*;

#[test]
fn each_site_has_the_documented_shape() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let cases = [
        ("residual AT LAYER 1", HIDDEN),
        ("post_attention AT LAYER 0", HIDDEN),
        ("ffn.input AT LAYER 1", HIDDEN),
        ("ffn.down_input AT LAYER 0", INTERMEDIATE),
        ("ffn.down_output AT LAYER 1", HIDDEN),
        ("final", HIDDEN),
        ("logits", VOCAB),
    ];
    for (i, (site, width)) in cases.iter().enumerate() {
        let out = fx.path(&format!("site{i}.npy"));
        capture(
            &mut s,
            &format!("{site} FOR \"{PROMPT}\" POSITIONS ALL"),
            &out,
        );
        let npy = read_npy(&out);
        assert_eq!(npy.shape, vec![PROMPT_TOKENS, *width], "{site}");
        assert_eq!(npy.values.len(), PROMPT_TOKENS * width, "{site}");
        assert!(npy.values.iter().all(|v| v.is_finite()), "{site}");
    }
}

#[test]
fn at_layers_stacks_one_block_per_layer_in_the_order_given() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let both = fx.path("both.npy");
    capture(
        &mut s,
        &format!("residual AT LAYERS (1, 0) FOR \"{PROMPT}\" POSITIONS (4, 0)"),
        &both,
    );
    let both = read_npy(&both);
    assert_eq!(both.shape, vec![2, 2, HIDDEN]);
    for (block, layer) in [(0, 1), (1, 0)] {
        let one = fx.path(&format!("layer{layer}.npy"));
        capture(
            &mut s,
            &format!("residual AT LAYER {layer} FOR \"{PROMPT}\" POSITIONS (4, 0)"),
            &one,
        );
        let one = read_npy(&one);
        assert_eq!(one.shape, vec![2, HIDDEN]);
        assert_eq!(bits(&one.values), bits(&rows(&both, block).concat()));
    }
    // A list of one layer is still a stacked array.
    let single = fx.path("single.npy");
    capture(
        &mut s,
        &format!("ffn.input AT LAYERS (1) FOR \"{PROMPT}\""),
        &single,
    );
    assert_eq!(read_npy(&single).shape, vec![1, 1, HIDDEN]);
}

#[test]
fn the_position_clauses_select_the_documented_positions() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let all = fx.path("all.npy");
    capture(
        &mut s,
        &format!("logits FOR \"{PROMPT}\" CONTINUE TOKENS (3, 4) POSITIONS ALL"),
        &all,
    );
    let all = rows(&read_npy(&all), 0);
    assert_eq!(all.len(), PROMPT_TOKENS + 2);
    let prompt_end = PROMPT_TOKENS - 1;
    let cases: [(&str, Vec<usize>); 5] = [
        ("PROMPT_END", vec![prompt_end]),
        ("REPLY", vec![prompt_end, prompt_end + 1]),
        ("LAST", vec![PROMPT_TOKENS + 1]),
        ("(6, 1, 3)", vec![6, 1, 3]),
        ("ALL", (0..PROMPT_TOKENS + 2).collect()),
    ];
    for (i, (clause, want)) in cases.iter().enumerate() {
        let out = fx.path(&format!("p{i}.npy"));
        capture(
            &mut s,
            &format!("logits FOR \"{PROMPT}\" CONTINUE TOKENS (3, 4) POSITIONS {clause}"),
            &out,
        );
        let got = rows(&read_npy(&out), 0);
        let expected: Vec<Vec<f32>> = want.iter().map(|&p| all[p].clone()).collect();
        assert_eq!(got.len(), expected.len(), "{clause}");
        for (g, e) in got.iter().zip(&expected) {
            assert_eq!(bits(g), bits(e), "{clause}");
        }
        let recorded: Vec<u64> = sidecar(&out)["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["position"].as_u64().unwrap())
            .collect();
        assert_eq!(
            recorded,
            want.iter().map(|&p| p as u64).collect::<Vec<_>>(),
            "{clause}"
        );
    }
    // Without a POSITIONS clause: the last position.
    let default = fx.path("default.npy");
    capture(&mut s, &format!("logits FOR \"{PROMPT}\""), &default);
    assert_eq!(read_npy(&default).shape, vec![1, VOCAB]);
}
