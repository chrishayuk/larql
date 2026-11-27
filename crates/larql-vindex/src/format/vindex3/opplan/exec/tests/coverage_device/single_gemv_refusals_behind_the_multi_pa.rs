//! Single-gemv refusals behind the multi path
//! FFN arms

use super::*;

/// On a kernelless device the single-gemv seam refuses each format
/// naming its kernel and the shape — the f16/MXFP4/NVFP4 arms a whole-
/// plan run never reaches because it dies at the multi-gemv first.
#[test]
fn single_gemv_refusals_name_the_kernel_and_shape_for_every_packed_format() {
    let backend = DevicePlanBackend::new(KernellessDevice, "kernelless-single", WeightFormat::F32);
    let f32_weight = lcg_values(ROWS * COLS, 8);
    let x = lcg_values(COLS, 9);
    let f16_bytes = vec![0u8; ROWS * COLS * F16_BYTES];
    let LoadedWeight::Mxfp4 {
        packed: mx_packed,
        scales: mx_scales,
    } = quantize_mxfp4(&f32_weight, ROWS, COLS, "mx").unwrap()
    else {
        unreachable!()
    };
    let LoadedWeight::Nvfp4 {
        packed: nv_packed,
        scales: nv_scales,
        tensor_scale,
        ..
    } = quantize_nvfp4(&f32_weight, ROWS, COLS, "nv").unwrap()
    else {
        unreachable!()
    };
    let cases: [(&str, WeightSlice<'_>); 3] = [
        ("f16_gemv", WeightSlice::F16(&f16_bytes)),
        (
            "mxfp4_gemv",
            WeightSlice::Mxfp4 {
                packed: mx_packed.as_slice(),
                scales: mx_scales.as_slice(),
            },
        ),
        (
            "nvfp4_gemv",
            WeightSlice::Nvfp4 {
                packed: nv_packed.as_slice(),
                scales: nv_scales.as_slice(),
                tensor_scale,
                activation: Default::default(),
            },
        ),
    ];
    for (kernel, weight) in cases {
        let message = project_on(&backend, weight, &x).unwrap_err().to_string();
        assert!(
            message.contains(&format!("device {kernel} [{ROWS} x {COLS}] refused")),
            "{kernel}: {message}"
        );
    }
}

/// A device that batches f16 matrices but has no single f16 kernel:
/// Q/K/V succeed through the multi path and the output projection —
/// a lone gemv — fails closed naming its own shape. Nothing widens or
/// substitutes.
#[test]
fn a_multi_only_f16_device_fails_closed_at_the_output_projection() {
    let (_c, plan, store) = dense_fixture();
    let backend = DevicePlanBackend::new(MultiOnlyF16Device, "multi-only-f16", WeightFormat::F16);
    let err = execute_plan(&plan, &store, &DENSE_TOKENS, &backend).unwrap_err();
    let message = err.to_string();
    let q_rows = super::super::Q_HEADS * super::super::HEAD_DIM;
    assert!(
        message.contains(&format!(
            "device f16_gemv [{} x {q_rows}] refused",
            super::super::HIDDEN
        )),
        "{message}"
    );
}

/// The NVFP4 multi path refuses on a kernelless device naming the
/// matrix count — the FFN's up+gate pair — and the FFN propagates it.
#[test]
fn an_nvfp4_multi_dispatch_refusal_names_the_matrix_count() {
    /// Up and gate.
    const FFN_PAIR: usize = 2;
    let backend = DevicePlanBackend::new(KernellessDevice, "kernelless-nvfp4", WeightFormat::Nvfp4);
    let values = lcg_values(FFN_INTERMEDIATE * FFN_HIDDEN, 10);
    let LoadedWeight::Nvfp4 {
        packed,
        scales,
        tensor_scale,
        ..
    } = quantize_nvfp4(&values, FFN_INTERMEDIATE, FFN_HIDDEN, "nv").unwrap()
    else {
        unreachable!()
    };
    let weight = WeightSlice::Nvfp4 {
        packed: packed.as_slice(),
        scales: scales.as_slice(),
        tensor_scale,
        activation: Default::default(),
    };
    let x = lcg_values(FFN_HIDDEN, 11);
    let err = backend
        .ffn(ffn_call(&x, Some(weight), weight, weight, Activation::Silu))
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains(&format!("nvfp4_gemv_multi ({FFN_PAIR} matrices) refused")),
        "{message}"
    );
}

/// The routed FFN's first device call is the f32 router; a kernelless
/// device refuses it and nothing routes.
#[test]
fn a_routed_ffn_whose_router_dispatch_is_refused_fails_closed() {
    let backend = DevicePlanBackend::new(KernellessDevice, "kernelless-routed", WeightFormat::F32);
    let router = lcg_values(EXPERTS * FFN_HIDDEN, 12);
    let expert_gate_up: Vec<Vec<f32>> = (0..EXPERTS)
        .map(|e| lcg_values(2 * FFN_INTERMEDIATE * FFN_HIDDEN, 20 + e as u64))
        .collect();
    let expert_down: Vec<Vec<f32>> = (0..EXPERTS)
        .map(|e| lcg_values(FFN_HIDDEN * FFN_INTERMEDIATE, 30 + e as u64))
        .collect();
    let gate_up: Vec<WeightSlice<'_>> =
        expert_gate_up.iter().map(|w| WeightSlice::F32(w)).collect();
    let down: Vec<WeightSlice<'_>> = expert_down.iter().map(|w| WeightSlice::F32(w)).collect();
    let x = lcg_values(FFN_HIDDEN, 13);
    let err = backend
        .routed_ffn(RoutedFfnCall {
            x: &x,
            hidden: FFN_HIDDEN,
            intermediate: FFN_INTERMEDIATE,
            experts: EXPERTS,
            top_k: TOP_K,
            router_kind: MoeRouterKind::TopKSoftmax,
            routing_policy: ExpertRoutingPolicy::SoftmaxThenSelect,
            branch_scale: 1.0,
            activation: Activation::Silu,
            gate_policy: ExpertGatePolicy::Gated,
            router: &router,
            router_bias: None,
            weights: ExpertSlices::Fused {
                gate_up: &gate_up,
                down: &down,
                layout: GateUpLayout::ContiguousHalves,
            },
            gate_up_bias: None,
            down_bias: None,
            router_input: None,
            router_scale: None,
            router_per_expert_scale: None,
            router_norm_eps: None,
        })
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains(&format!(
            "device f32_gemv [{EXPERTS} x {FFN_HIDDEN}] refused"
        )),
        "{message}"
    );
}

/// An ungated FFN is `down · silu(up · x)`: no gate matrix, no
/// elementwise product — the shape the dense-Llama plan never carries.
#[test]
fn an_ungated_ffn_applies_silu_to_the_up_projection_alone() {
    let backend = DevicePlanBackend::new(LoopDevice, "loop-device-ungated", WeightFormat::F32);
    let up = lcg_values(FFN_INTERMEDIATE * FFN_HIDDEN, 14);
    let down = lcg_values(FFN_HIDDEN * FFN_INTERMEDIATE, 15);
    let x = lcg_values(FFN_HIDDEN, 16);
    let got = backend
        .ffn(ffn_call(
            &x,
            None,
            WeightSlice::F32(&up),
            WeightSlice::F32(&down),
            Activation::Silu,
        ))
        .unwrap();
    let inner: Vec<f32> = matvec_rows(&up, FFN_INTERMEDIATE, FFN_HIDDEN, &x)
        .into_iter()
        .map(silu)
        .collect();
    let expected = matvec_rows(&down, FFN_HIDDEN, FFN_INTERMEDIATE, &inner);
    assert_eq!(got.len(), FFN_HIDDEN);
    assert!(
        max_abs(&got, &expected) < LOOP_NOISE_CEILING,
        "ungated FFN diverges from down · silu(up · x)"
    );
}

/// The device FFN refuses an activation it has no kernel for, in both
/// the gated and the ungated shape, naming which shape refused.
#[test]
fn an_ffn_with_an_unsupported_activation_refuses_gated_and_ungated() {
    let backend = DevicePlanBackend::new(LoopDevice, "loop-device-activation", WeightFormat::F32);
    let up = lcg_values(FFN_INTERMEDIATE * FFN_HIDDEN, 17);
    let down = lcg_values(FFN_HIDDEN * FFN_INTERMEDIATE, 18);
    let x = lcg_values(FFN_HIDDEN, 19);
    for (shape, gate) in [("gated", Some(WeightSlice::F32(&up))), ("ungated", None)] {
        let err = backend
            .ffn(ffn_call(
                &x,
                gate,
                WeightSlice::F32(&up),
                WeightSlice::F32(&down),
                Activation::Gelu,
            ))
            .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains(&format!("{shape}-FFN")) && message.contains("Gelu"),
            "{shape}: {message}"
        );
    }
}

/// The device backend does not lower FFN projection biases; a plan that
/// carries them is refused by name, never run unbiased.
#[test]
fn ffn_projection_biases_are_refused_by_name() {
    let backend = DevicePlanBackend::new(LoopDevice, "loop-device-biased", WeightFormat::F32);
    let up = lcg_values(FFN_INTERMEDIATE * FFN_HIDDEN, 21);
    let down = lcg_values(FFN_HIDDEN * FFN_INTERMEDIATE, 22);
    let x = lcg_values(FFN_HIDDEN, 23);
    let bias = vec![0.5; FFN_INTERMEDIATE];
    let mut call = ffn_call(
        &x,
        None,
        WeightSlice::F32(&up),
        WeightSlice::F32(&down),
        Activation::GeluTanh,
    );
    call.up_bias = Some(&bias);
    let err = backend.ffn(call).unwrap_err().to_string();
    assert!(
        err.contains("FFN projection biases are not lowered"),
        "{err}"
    );
}
