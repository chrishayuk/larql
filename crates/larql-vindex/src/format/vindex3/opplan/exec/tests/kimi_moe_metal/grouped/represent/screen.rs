//! The quality half of Q1, and the controls it depends on.

use super::*;

/// **The screen.** Every candidate, end to end, against the
/// checkpoint's own expert outputs.
#[test]
fn report_expert_bank_representation_screen() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let oracle: Vec<f32> = fx.experts.iter().flat_map(|e| e.oracle.clone()).collect();
    // Every arm built and HELD before any dispatch — see [`Arm`].
    let arms: Vec<Arm> = Format::ALL.iter().map(|&f| Arm::build(&fx, f)).collect();
    let outputs: Vec<Vec<f32>> = arms.iter().map(|a| a.ffn(&metal, &fx)).collect();

    let bf16 = &outputs[0];
    let floor_rms = rel_rms(bf16, &oracle);
    eprintln!(
        "[q1a] {} slots at Kimi's real expert geometry (hidden {}, inter {}); the BF16 arm \
         is the floor, not zero: rel_rms {floor_rms:.3e} against modeling_kimi.py's own outputs",
        fx.experts.len(),
        fx.hidden,
        fx.inter
    );
    eprintln!(
        "[q1a] {:<6} {:>7} {:>9} {:>11} {:>11} {:>8}  {}",
        "format", "bpw", "MiB/bank", "rel_rms", "max/scale", "x floor", "dispatch"
    );
    for (a, out) in arms.iter().zip(&outputs) {
        let rms = rel_rms(out, &oracle);
        eprintln!(
            "[q1a] {:<6} {:>7.3} {:>9.1} {:>11.3e} {:>11.3e} {:>8.0}  {}",
            a.format.label(),
            a.format.bpw(),
            a.bank_bytes() as f64 / (1024.0 * 1024.0),
            rms,
            rel_err(out, &oracle),
            rms / floor_rms,
            if a.format.native() {
                "native kernel"
            } else {
                "simulated (bf16 carrier)"
            }
        );
        assert!(
            rms < 1.0,
            "{}: rel_rms {rms} means the bank carries no signal",
            a.format.label()
        );
    }

    // The BF16 arm must reproduce itself exactly, so any movement in the
    // table is the representation and not the harness. Run against the
    // SAME held bank, which is the condition that makes it meaningful.
    assert_eq!(
        arms[0].ffn(&metal, &fx),
        *bf16,
        "the BF16 arm must be deterministic"
    );
}

/// **The control the screen depends on: the encoder and the Metal
/// kernel must agree about the bytes.**
///
/// ADR-008 exists because a build tool once reimplemented Q4_K with
/// different formulas, producing banks that were internally consistent
/// and wrong through the shaders. A quantisation error and a layout
/// disagreement look identical from a single arm — both read as "the
/// quantised answer moved" — so the screen means nothing until the SAME
/// bytes are shown to give the SAME answer through an independent
/// consumer that shares no code with the shader.
#[test]
fn the_cpu_and_metal_consumers_agree_about_the_same_quantised_bytes() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let e = &fx.experts[0];

    // BOTH stages: the projections have different `k` (2304 and 1024) and
    // the down stage is the per-slot one. A control that only covered
    // gate would pass while the shape the FFN actually ends on was
    // broken.
    // Encoded up front and HELD: `q6k_grouped_experts` binds its weights
    // through `get_bytes`, which caches on `(ptr, len)`, and these four
    // banks come in two identical sizes — so encoding one per iteration
    // lets the down bank land on the gate bank's cached address. That
    // failed 2 runs in 4 here before the banks were hoisted.
    let cases: Vec<(Stage, Format, Vec<u8>)> = [
        (Stage::Gate, Format::Q6K),
        (Stage::Gate, Format::Q4K),
        (Stage::Down, Format::Q6K),
        (Stage::Down, Format::Q4K),
    ]
    .into_iter()
    .map(|(stage, f)| {
        let (_, k) = stage.shape(&fx);
        (stage, f, f.encode(stage.matrix(e), k))
    })
    .collect();

    for (stage, f, coded) in &cases {
        let (stage, f) = (*stage, *f);
        let (n, k) = stage.shape(&fx);
        let x: Vec<f32> = match stage {
            Stage::Down => fx.x.iter().cycle().take(k).copied().collect(),
            _ => fx.x.clone(),
        };
        let device = f.dispatch(
            &metal,
            coded,
            &[ExpertOffset(0)],
            &x,
            n,
            k,
            InputLayout::Shared,
        );
        let mut host = vec![0.0f32; n];
        match f {
            Format::Q6K => q6k_matmul_into(&mut host, &x, coded, n, k, 1),
            Format::Q4K => q4k_matmul_into(&mut host, &x, coded, n, k, 1),
            Format::Bf16 | Format::Mxfp4 => unreachable!("only k-quants are checked here"),
        }
        let finite = device.iter().all(|v| v.is_finite());
        let disagreement = if finite {
            rel_rms(&device, &host)
        } else {
            f32::NAN
        };
        eprintln!(
            "[q1a-control] {} {} (n={n}, k={k}): Metal grouped vs larql_compute CPU matmul \
             on the SAME bytes — rel_rms {disagreement:.3e}, device finite {finite}, \
             host finite {}",
            f.label(),
            stage.label(),
            host.iter().all(|v| v.is_finite())
        );
        assert!(
            finite,
            "{} {}: the Metal kernel produced non-finite values",
            f.label(),
            stage.label()
        );
        assert!(
            disagreement < 1e-3,
            "{} {}: the Metal kernel and the CPU decoder disagree by {disagreement:.3e} on \
             identical bytes — a LAYOUT defect, and every number in the screen would be \
             measuring it rather than the format",
            f.label(),
            stage.label()
        );
    }
    let (n, k) = Stage::Gate.shape(&fx);

    // And the control that this control can fail: bytes encoded as one
    // format, read as the other, must NOT agree. "Different" has to
    // include "not a number" — decoding 210-byte superblocks as 144-byte
    // ones reads scale fields out of arbitrary payload, and a NaN makes
    // every ordinary comparison false, so a naive `>` would report the
    // control as failing to fail.
    let as_q6 = Format::Q6K.encode(Stage::Gate.matrix(e), k);
    let mut misread = vec![0.0f32; n];
    q4k_matmul_into(&mut misread, &fx.x, &as_q6, n, k, 1);
    let mut correct = vec![0.0f32; n];
    q6k_matmul_into(&mut correct, &fx.x, &as_q6, n, k, 1);
    assert!(
        misread
            .iter()
            .zip(&correct)
            .any(|(a, b)| !a.is_finite() || (a - b).abs() > 1e-3 * b.abs().max(1e-6)),
        "reading Q6_K bytes as Q4_K must produce a different answer, or the agreement \
         above is not evidence of anything"
    );
}

/// **What simulating a format costs, measured rather than argued.**
///
/// MXFP4 is screened through decoded values carried in bf16, because its
/// grouped kernel is not on this convention. That substitution is only
/// legitimate if the carrier's own error is small against the error
/// being measured — so it is measured here, on the two formats where
/// both arms exist.
///
/// The finding is why the screen dispatches natively wherever it can:
/// the bf16 carrier is adequate for the 4-bit class and marginal at
/// 6-bit.
#[test]
fn report_what_the_simulation_costs() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let e = &fx.experts[0];
    let (n, k) = Stage::Gate.shape(&fx);
    let exact = Format::Bf16.dispatch(
        &metal,
        Stage::Gate.matrix(e),
        &[ExpertOffset(0)],
        &fx.x,
        n,
        k,
        InputLayout::Shared,
    );

    for f in [Format::Q6K, Format::Q4K] {
        // Both banks alive at once, for the reason [`Arm`] documents.
        let native_bank = f.encode(Stage::Gate.matrix(e), k);
        let sim_bank = f.simulate(Stage::Gate.matrix(e), n, k);
        let native = f.dispatch(
            &metal,
            &native_bank,
            &[ExpertOffset(0)],
            &fx.x,
            n,
            k,
            InputLayout::Shared,
        );
        let simulated = Format::Bf16.dispatch(
            &metal,
            &sim_bank,
            &[ExpertOffset(0)],
            &fx.x,
            n,
            k,
            InputLayout::Shared,
        );
        let signal = rel_rms(&native, &exact);
        let carrier = rel_rms(&simulated, &native);
        eprintln!(
            "[q1a-sim] {}: format error vs BF16 {signal:.3e}, simulation-vs-native \
             {carrier:.3e} ({:.1}x smaller)",
            f.label(),
            signal / carrier
        );
        assert!(
            carrier < signal,
            "{}: the bf16 carrier costs {carrier:.3e} against a format error of only \
             {signal:.3e} — simulation would be reporting the carrier",
            f.label()
        );
    }
}

/// **The grouped contract for the k-quants: many slots, and the
/// per-slot input layout.**
///
/// The single-expert control above pins the ENCODING against an
/// independent decoder. This pins the GROUPING: nine slots addressed
/// through a byte-offset table, and the down projection's per-slot
/// input, which is the one layout the FFN ends on and the one a
/// single-slot control cannot reach.
#[test]
fn the_k_quant_grouped_dispatch_is_correct_across_slots_and_layouts() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    // EVERY bank built and held before any dispatch. `q6k_grouped_experts`
    // binds its weights through `get_bytes`, which caches on `(ptr, len)`
    // — and these four banks come in two identical sizes, so building one
    // per iteration lets the allocator hand the down bank the address the
    // gate bank's buffer is still cached under. That reproduced as "Q6_K
    // per-slot is broken" in 2 runs of 5.
    let cases: Vec<(Format, Stage, Reencoded)> = [Format::Q6K, Format::Q4K]
        .into_iter()
        .flat_map(|f| [Stage::Gate, Stage::Down].into_iter().map(move |s| (f, s)))
        .map(|(f, stage)| (f, stage, Reencoded::build(&fx, stage, f)))
        .collect();
    let slots = fx.experts.len();

    for (f, stage, bank) in &cases {
        let (f, stage) = (*f, *stage);
        let (n, k) = stage.shape(&fx);
        let layout = if stage == Stage::Down {
            InputLayout::PerSlot
        } else {
            InputLayout::Shared
        };
        // Distinct values per slot, so a kernel that ignored the stride
        // and gave every slot slot 0's vector would be caught.
        let x: Vec<f32> = match layout {
            InputLayout::Shared => fx.x.clone(),
            InputLayout::PerSlot => fx.x.iter().cycle().take(k * slots).copied().collect(),
        };
        let got = f.dispatch(&metal, &bank.bytes, &bank.offsets, &x, n, k, layout);

        // Slot by slot on the CPU, from the same bank at the same
        // offsets — so a table the kernel reads differently shows up as
        // one slot's answer appearing under another's index.
        let per_expert = bank.bytes.len() / slots;
        let mut want = vec![0.0f32; n * slots];
        for s in 0..slots {
            let payload = &bank.bytes[s * per_expert..(s + 1) * per_expert];
            let xs = match layout {
                InputLayout::Shared => &x[..],
                InputLayout::PerSlot => &x[s * k..(s + 1) * k],
            };
            let out = &mut want[s * n..(s + 1) * n];
            match f {
                Format::Q6K => q6k_matmul_into(out, xs, payload, n, k, 1),
                Format::Q4K => q4k_matmul_into(out, xs, payload, n, k, 1),
                Format::Bf16 | Format::Mxfp4 => unreachable!("only k-quants here"),
            }
        }

        let finite = got.iter().all(|v| v.is_finite());
        let overall = if finite {
            rel_rms(&got, &want)
        } else {
            f32::NAN
        };
        eprintln!(
            "[q1a-grouped] {} {} slots={slots} layout={} — finite {finite}, rel_rms {overall:.3e}",
            f.label(),
            stage.label(),
            if layout == InputLayout::PerSlot {
                "per-slot"
            } else {
                "shared"
            }
        );
        assert!(
            finite,
            "{} {}: the grouped dispatch produced non-finite values across {slots} slots",
            f.label(),
            stage.label()
        );
        if !(overall < 1e-3) {
            for s in 0..slots {
                eprintln!(
                    "[q1a-grouped]   slot {s}: vs own {:.3e}, vs slot0 {:.3e}",
                    rel_rms(&got[s * n..(s + 1) * n], &want[s * n..(s + 1) * n]),
                    rel_rms(&got[s * n..(s + 1) * n], &want[..n])
                );
            }
        }
        assert!(
            overall < 1e-3,
            "{} {}: grouped disagrees with slot-by-slot on the same bank",
            f.label(),
            stage.label()
        );
    }
}
