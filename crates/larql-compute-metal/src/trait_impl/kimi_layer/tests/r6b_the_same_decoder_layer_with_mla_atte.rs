//! R6b: the same decoder layer with MLA attention

use super::*;

/// **R6b.** The decoder layer with MLA attention instead of KDA, against
/// a reference composed of its already-gated parts.
///
/// The layer path is written once and takes the attention as a
/// parameter, so this checks that the MLA arm binds the same way — and
/// that the cache advances exactly once per layer call, which is the one
/// thing a chained encode can get wrong (the latent is only really
/// cached once the dispatch that wrote it has run).
#[test]
fn an_mla_decoder_layer_matches_a_reference_composed_of_its_parts() {
    let m = backend();
    let f = fixture();
    let bits = mla_bits();
    let state = MlaDeviceState::with_capacity(&m, mla_shape(), 8);

    fn mla_layer<'a>(
        f: &'a Fixture,
        bits: &'a MlaBits,
        st: &'a MlaDeviceState,
    ) -> KimiLayerWeights<'a> {
        KimiLayerWeights {
            input_norm: &f.input_norm,
            post_attention_norm: &f.post_norm,
            attention: AttentionSpec::Mla {
                weights: bits.device(),
                shape: mla_shape(),
                state: st,
            },
            ffn: FfnSpec::Moe(KimiMoeWeights {
                router_weight: &f.router_weight,
                router_bias: &f.router_bias,
                gate: projection(&f.bank_gate, &f.residency, &f.shared_gate),
                up: projection(&f.bank_up, &f.residency, &f.shared_up),
                down: projection(&f.bank_down, &f.residency, &f.shared_down),
                inter: INTER,
                top_k: TOP_K,
                renormalize: true,
                branch_scale: BRANCH_SCALE,
            }),
            norm_eps: EPS,
        }
    }

    // Two positions, so the cache is genuinely read on the second.
    let ref_state = MlaDeviceState::with_capacity(&m, mla_shape(), 8);
    for pos in 0..2 {
        let x: Vec<f32> = f.x.iter().map(|v| v + pos as f32 * 0.13).collect();
        let got = m
            .kimi_decoder_layer_traced(mla_layer(&f, &bits, &state), &x)
            .expect("mla decoder layer");
        assert_eq!(
            state.len(),
            pos + 1,
            "the MLA cache must advance exactly once a layer call"
        );

        // The reference: the gated attention, then the host's own
        // residual / norm / router / combine.
        let normed = rms_norm(&x, &f.input_norm, EPS);
        let (attn, _) = m
            .mla_attention_step(bits.device(), mla_shape(), &ref_state, &normed)
            .expect("attention alone");
        let after: Vec<f32> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();
        let post = rms_norm(&after, &f.post_norm, EPS);
        let (ids, weights) = route(&post, &f.router_weight, &f.router_bias);

        assert!(
            max_abs(&got.attention, &attn) < TOLERANCE,
            "pos {pos} attention"
        );
        assert!(
            max_abs(&got.after_attention, &after) < TOLERANCE,
            "pos {pos} residual"
        );
        assert!(
            max_abs(&got.post_attention_normed, &post) < TOLERANCE,
            "pos {pos} post norm"
        );
        assert_eq!(
            got.selected_ids,
            ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
            "pos {pos}: the device chose other experts"
        );
        assert!(
            max_abs(&got.combine_weights[..TOP_K], &weights) < TOLERANCE,
            "pos {pos} routed weights"
        );

        let offsets: Vec<ExpertOffset> =
            ids.iter().map(|&i| ExpertOffset(f.residency[i])).collect();
        let (outs, _) = m
            .bf16_moe_ffn_blocks(
                &[MoeBlockCall {
                    banks: MoeFfnBanks {
                        gate: bank(&f.bank_gate, &offsets),
                        up: bank(&f.bank_up, &offsets),
                        down: bank(&f.bank_down, &offsets),
                        hidden: HIDDEN,
                        inter: INTER,
                    },
                    x: &post,
                }],
                BlockLowering::Separate,
            )
            .expect("experts");
        let zero = [ExpertOffset(0)];
        let (shared_outs, _) = m
            .bf16_moe_ffn_blocks(
                &[MoeBlockCall {
                    banks: MoeFfnBanks {
                        gate: bank(&f.shared_gate, &zero),
                        up: bank(&f.shared_up, &zero),
                        down: bank(&f.shared_down, &zero),
                        hidden: HIDDEN,
                        inter: INTER,
                    },
                    x: &post,
                }],
                BlockLowering::Separate,
            )
            .expect("shared expert");
        let all_outs: Vec<f32> = outs[0].iter().chain(&shared_outs[0]).copied().collect();
        let mut want = after.clone();
        for (slot, w) in got.combine_weights.iter().enumerate() {
            for (j, o) in want.iter_mut().enumerate() {
                *o += w * all_outs[slot * HIDDEN + j];
            }
        }
        assert!(
            max_abs(&got.output, &want) < TOLERANCE,
            "pos {pos} layer output"
        );
    }
}

/// A KDA layer and an MLA layer in ONE command buffer — the shape R6c
/// runs at real weights. Both caches must advance, and the second layer
/// must read the first's output.
#[test]
fn a_kda_layer_and_an_mla_layer_share_one_command_buffer() {
    let m = backend();
    let f = fixture();
    let bits = mla_bits();
    let kda_state = KdaDeviceState::zeros(&m, shape());
    let mla_state = MlaDeviceState::with_capacity(&m, mla_shape(), 8);

    let mut second = f.layer(&kda_state);
    second.attention = AttentionSpec::Mla {
        weights: bits.device(),
        shape: mla_shape(),
        state: &mla_state,
    };
    let chain = [
        KimiLayerCall {
            weights: f.layer(&kda_state),
        },
        KimiLayerCall { weights: second },
    ];
    let planes = m
        .kimi_decoder_layers_traced(&chain, &f.x)
        .expect("mixed chain");
    assert_eq!(planes.len(), 2);
    assert_eq!(mla_state.len(), 1, "the MLA cache advanced once");

    // Layer 1 read layer 0's OUTPUT, not the original input: its input
    // norm is the norm OF that output, which a layer fed the original
    // `x` could not produce.
    let want = rms_norm(&planes[0].output, &f.input_norm, EPS);
    assert!(
        max_abs(&planes[1].input_normed, &want) < TOLERANCE,
        "layer 1's input norm is not the norm of layer 0's output"
    );
    let wrong = rms_norm(&f.x, &f.input_norm, EPS);
    assert!(
        max_abs(&planes[1].input_normed, &wrong) > TOLERANCE,
        "control: the two candidate inputs must be distinguishable"
    );
}

/// **A failed command buffer refuses the whole chain and advances nothing.**
///
/// The fault is injected at the chain's own wait site, so it fires after
/// the (valid) buffer ran; what is under test is the host's response: a
/// typed refusal naming the site, no readback, the MLA cache still at its
/// pre-call length, and a backend that serves the next call normally.
/// This is the path the Kimi MLA/KDA trajectory evidence runs through.
#[test]
fn a_failed_command_buffer_refuses_the_chain_and_advances_nothing() {
    let m = backend();
    let f = fixture();
    let bits = mla_bits();
    let kda_state = KdaDeviceState::zeros(&m, shape());
    let mla_state = MlaDeviceState::with_capacity(&m, mla_shape(), 8);

    let mut second = f.layer(&kda_state);
    second.attention = AttentionSpec::Mla {
        weights: bits.device(),
        shape: mla_shape(),
        state: &mla_state,
    };
    let chain = [
        KimiLayerCall {
            weights: f.layer(&kda_state),
        },
        KimiLayerCall { weights: second },
    ];

    crate::cb_status::inject_fault_at_for_test("kimi_layer/mod.rs:layers");
    let faults_before = crate::cb_status::non_completed_count();
    let err = m
        .kimi_decoder_layers(&chain, &f.x, None)
        .expect_err("a failed command buffer must refuse the chain");
    assert!(
        matches!(&err, GroupedError::CommandBufferFailed { site, .. }
            if site.ends_with("kimi_layer/mod.rs:layers")),
        "refusal must name the chain's wait site: {err}"
    );
    assert_eq!(
        mla_state.len(),
        0,
        "the MLA cache must not advance past a failed buffer"
    );
    assert!(
        !crate::cb_status::injected_fault_pending(),
        "the fault fired at the chain's own wait, not at an earlier one"
    );
    assert_eq!(
        crate::cb_status::non_completed_count(),
        faults_before,
        "an injected fault is not a GPU observation"
    );

    // The refusal left the backend usable: the same chain now runs and
    // advances the cache exactly once.
    let (out, _) = m
        .kimi_decoder_layers(&chain, &f.x, None)
        .expect("a healthy chain after a refused one");
    assert_eq!(out.len(), f.x.len());
    assert_eq!(
        mla_state.len(),
        1,
        "the healthy call advanced the cache once"
    );
}

/// **The route trace reports what the router actually decided.**
///
/// Checked against the host reference rather than against itself, and
/// against the traced path's own reading of the same buffer — so the
/// cheap instrumentation is pinned to the expensive one that is already
/// gated, not merely to itself.
#[test]
fn the_route_trace_matches_the_routers_own_decision() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let f = fixture();
    let state = KdaDeviceState::zeros(&b, shape());
    let calls = [KimiLayerCall {
        weights: f.layer(&state),
    }];

    let mut trace = ExecutionTrace::default();
    let (out, _) = b
        .kimi_decoder_layers(&calls, &f.x, Some(&mut trace))
        .expect("chain runs");
    assert_eq!(trace.routes.len(), 1, "one entry a layer");
    assert_eq!(trace.routes[0].len(), TOP_K);

    // The host reference for the same input.
    let normed = rms_norm(&f.x, &f.input_norm, EPS);
    let attention = {
        let s2 = KdaDeviceState::zeros(&b, shape());
        let p = b
            .kimi_decoder_layer_traced(f.layer(&s2), &f.x)
            .expect("traced");
        p.attention
    };
    let after: Vec<f32> = f.x.iter().zip(&attention).map(|(a, c)| a + c).collect();
    let post = rms_norm(&after, &f.post_norm, EPS);
    let (want_ids, _) = route(&post, &f.router_weight, &f.router_bias);
    let want: Vec<u32> = want_ids.iter().map(|i| *i as u32).collect();
    assert_eq!(
        trace.routes[0], want,
        "the trace must carry the router's OWN selection, in router order"
    );
    let _ = normed;

    // Serving passes `None`, and must get the same answer — the
    // instrumentation may not perturb what it observes.
    let state_b = KdaDeviceState::zeros(&b, shape());
    let (untraced, _) = b
        .kimi_decoder_layers(
            &[KimiLayerCall {
                weights: f.layer(&state_b),
            }],
            &f.x,
            None,
        )
        .expect("untraced");
    assert_eq!(untraced, out, "tracing must not change the answer");
}

/// **Identity addressing is the same arithmetic, not a second path.**
///
/// A full bank whose experts sit at their own index must give exactly
/// what a table spelling out those offsets gives — so `Identity` is a
/// way of NOT tabulating an address, never a different lowering.
///
/// The fixture's bank is packed, so the comparison is made on the one
/// shape where both can describe the same thing: a table that happens to
/// be the identity map.
#[test]
fn identity_addressing_equals_a_table_that_spells_out_the_same_offsets() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let f = fixture();
    let stride = (INTER * HIDDEN * 2) as u32;
    // A bank holding every scored expert at its own index.
    let per = INTER * HIDDEN * 2;
    let full_gate: Vec<u8> = (0..EXPERTS)
        .flat_map(|e| bf16_bytes(INTER, HIDDEN, 3.0 + e as f32))
        .collect();
    let full_up: Vec<u8> = (0..EXPERTS)
        .flat_map(|e| bf16_bytes(INTER, HIDDEN, 9.0 + e as f32))
        .collect();
    let full_down: Vec<u8> = (0..EXPERTS)
        .flat_map(|e| bf16_bytes(HIDDEN, INTER, 15.0 + e as f32))
        .collect();
    assert_eq!(full_gate.len(), EXPERTS * per);
    let spelled: Vec<u32> = (0..EXPERTS).map(|e| (e * per) as u32).collect();

    let run = |addressing: ExpertAddressing<'_>| {
        let state = KdaDeviceState::zeros(&b, shape());
        let mut w = f.layer(&state);
        // The shared branch is its own region in both arms — identical,
        // so any difference is routed addressing.
        fn region(bytes: &[u8]) -> EncodedRegion<'_> {
            EncodedRegion {
                bytes,
                encoding: ExpertEncoding::Bf16,
            }
        }
        let m = KimiMoeWeights {
            router_weight: &f.router_weight,
            router_bias: &f.router_bias,
            gate: ProjectionBank {
                routed: region(&full_gate),
                addressing,
                shared: Some(region(&f.shared_gate)),
            },
            up: ProjectionBank {
                routed: region(&full_up),
                addressing,
                shared: Some(region(&f.shared_up)),
            },
            down: ProjectionBank {
                routed: region(&full_down),
                addressing,
                shared: Some(region(&f.shared_down)),
            },
            inter: INTER,
            top_k: TOP_K,
            renormalize: true,
            branch_scale: BRANCH_SCALE,
        };
        w.ffn = FfnSpec::Moe(m);
        b.kimi_decoder_layer(w, &f.x).expect("runs").0
    };

    let identity = run(ExpertAddressing::Identity {
        experts: EXPERTS,
        stride,
    });
    let tabulated = run(ExpertAddressing::Table(&spelled));
    assert_eq!(
        identity, tabulated,
        "identity addressing must be the same arithmetic as a table saying the same thing"
    );

    // And a full bank cannot refuse: every scored expert is addressable,
    // so there is no NOT_RESIDENT to hit however the router routes.
    let a = ExpertAddressing::Identity {
        experts: EXPERTS,
        stride,
    };
    for e in 0..EXPERTS {
        assert_eq!(a.offset_of(e), Some((e * per) as u64));
    }
    assert_eq!(
        a.offset_of(EXPERTS),
        None,
        "outside the bank is still refused"
    );
}

/// A refused MLA layer call advances nothing.
///
/// The route is refused after the dispatch has run, because that is when
/// the refusal counter can be read. The call returns no output, so the
/// attention cache must not keep the position: it used to advance before
/// the refusal was checked, and a caller that handled the refusal and
/// tried again was one position ahead of its own stream.
#[test]
fn a_refused_mla_layer_call_leaves_the_cache_where_it_was() {
    let m = backend();
    let f = fixture();
    let bits = mla_bits();
    let state = MlaDeviceState::with_capacity(&m, mla_shape(), 8);
    let layer = |residency| KimiLayerWeights {
        input_norm: &f.input_norm,
        post_attention_norm: &f.post_norm,
        attention: AttentionSpec::Mla {
            weights: bits.device(),
            shape: mla_shape(),
            state: &state,
        },
        ffn: FfnSpec::Moe(KimiMoeWeights {
            router_weight: &f.router_weight,
            router_bias: &f.router_bias,
            gate: projection(&f.bank_gate, residency, &f.shared_gate),
            up: projection(&f.bank_up, residency, &f.shared_up),
            down: projection(&f.bank_down, residency, &f.shared_down),
            inter: INTER,
            top_k: TOP_K,
            renormalize: true,
            branch_scale: BRANCH_SCALE,
        }),
        norm_eps: EPS,
    };
    // Evict every resident expert but one, so the route must leave the bank.
    let mut evicted = f.residency.clone();
    for slot in evicted.iter_mut().take(RESIDENT).skip(1) {
        *slot = layer_shader::NOT_RESIDENT;
    }

    let refused = m.kimi_decoder_layer(layer(&evicted), &f.x);

    assert!(
        matches!(
            refused,
            Err(GroupedError::LayerRouteNotResident { layer: 0, .. })
        ),
        "the evicted route must be refused"
    );
    assert_eq!(state.len(), 0, "a refused call must not advance the cache");

    // The control: the same layer with its residency intact runs, and that
    // call is the one that advances the cache.
    m.kimi_decoder_layer(layer(&f.residency), &f.x)
        .expect("the resident route runs");
    assert_eq!(state.len(), 1);
}
