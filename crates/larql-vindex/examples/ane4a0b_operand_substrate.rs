//! ANE-4A0b — does a reduced physical plan cost what the PLAN costs, or
//! what the CONTAINER costs?
//!
//! The invariant under test:
//!
//! > The cost of executing a reduced physical plan scales with the plan,
//! > not with the size of the authoritative container.
//!
//! That is what makes `ExecutionSlice` a physical *view* over VINDEX3
//! rather than a filter applied after the fact. A four-layer draft that
//! reads sixty-four layers of weights is not a view; it is a full model
//! that throws most of its work away, and it would make every
//! reduced-depth measurement meaningless as an economics claim.
//!
//! Reading the code says the invariant holds — `OperandStore` seeks to
//! one tensor per operand, and `PreparedOperands::load` slices the layer
//! range before loading anything. This measures it instead, because the
//! nastier failure mode is exactly the one reading misses: execution
//! skipping layers while preparation still reads their weights.
//!
//! Two instruments, and they answer different questions:
//!
//! ```text
//! load_count   how many operands were READ from the container
//!              -> the view claim
//! peak RSS     how much was MATERIALISED as f32 at once
//!              -> the affordability claim
//! ```
//!
//! `load_count` is the one that can prove the invariant; RSS can be
//! confounded by allocator behaviour and page cache, so it is reported
//! rather than asserted on. Take RSS from the outside:
//!
//! ```text
//! /usr/bin/time -l cargo run --release -p larql-vindex \
//!     --example ane4a0b_operand_substrate -- <container> 1,2,4
//! ```
//!
//! Deliberately does NOT run `Full` on a 27B container: that is ~93 GB of
//! f32 on a machine whose disk cannot grow swap. The depths here are
//! chosen to stay bounded, and the point is the SHAPE of the curve.

use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::execute_slice;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::ExecutionSlice;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::plan_component_ops;
use std::time::Instant;

/// Any valid ids: this rung measures the operand substrate, not the
/// model's answer. Real tokenisation matters at the depth ladder, not
/// here.
const TOKENS: &[u32] = &[1, 2, 3];

fn main() {
    let mut args = std::env::args().skip(1);
    let container = args.next().unwrap_or_else(|| {
        eprintln!("usage: ane4a0b_operand_substrate <container> [depths, default 1,2,4]");
        std::process::exit(2);
    });
    let depths: Vec<usize> = args
        .next()
        .unwrap_or_else(|| "1,2,4".to_string())
        .split(',')
        .map(|d| d.trim().parse().expect("depth"))
        .collect();

    let root = std::path::Path::new(&container);
    let inspection = inspect_container(root, false).expect("inspect");
    let plan = plan_component_ops(&inspection, root, "target")
        .expect("plan")
        .plan
        .expect("the container carries a target plan");
    let depth = plan.layers.len();
    println!("container: {container}");
    println!("component `{}` has {depth} layers\n", plan.component);

    println!(
        "{:>7}{:>10}{:>14}{:>16}{:>12}",
        "depth", "layers", "operand reads", "reads/layer", "wall s"
    );

    let mut rows: Vec<(usize, u64)> = Vec::new();
    for &end in &depths {
        if end > depth {
            eprintln!("skipping depth {end}: deeper than the model");
            continue;
        }
        // A fresh store per arm, so `load_count` is this arm's reads and
        // not a running total across arms.
        let store = OperandStore::open(root, &inspection).expect("store");
        let before = store.load_count();
        let t = Instant::now();
        let trace = execute_slice(
            &plan,
            &store,
            TOKENS,
            &ReferenceBackend,
            ExecutionSlice::Draft { end },
        )
        .expect("draft traversal");
        let secs = t.elapsed().as_secs_f64();
        let reads = store.load_count() - before;

        // The guard: execution must have run exactly the requested
        // prefix. Without this, a low read count could mean the slice
        // worked or that the traversal quietly did less than it claimed.
        assert_eq!(
            trace.executed_layers,
            (0..end).collect::<Vec<_>>(),
            "draft of depth {end} executed {:?}",
            trace.executed_layers
        );
        assert!(
            trace.logits.as_ref().is_some_and(|l| !l.is_empty()),
            "a draft owns the head, so it must produce logits"
        );

        println!(
            "{:>7}{:>10}{:>14}{:>16.1}{:>12.2}",
            end,
            trace.executed_layers.len(),
            reads,
            reads as f64 / end as f64,
            secs
        );
        rows.push((end, reads));
    }

    // The invariant, stated as arithmetic rather than left to the eye.
    //
    // Reads should be affine in depth: a fixed cost for the ends
    // (embedding table, final norm, output head) plus a constant per
    // layer. If reads were instead flat in depth, preparation is reading
    // the whole container regardless of the slice, and the "view" claim
    // is false.
    if rows.len() >= 2 {
        let (d0, r0) = rows[0];
        let (d1, r1) = rows[rows.len() - 1];
        let per_layer = (r1 as f64 - r0 as f64) / (d1 as f64 - d0 as f64);
        let ends = r0 as f64 - per_layer * d0 as f64;
        println!("\nreads ~ {ends:.0} + {per_layer:.1} per layer");
        println!(
            "extrapolated to the full {depth}-layer model: {:.0} reads",
            ends + per_layer * depth as f64
        );
        if per_layer <= 0.0 {
            println!(
                "\nINVARIANT VIOLATED: reads do not grow with depth — preparation is \
                 not honouring the slice"
            );
            std::process::exit(1);
        }
        println!(
            "\nINVARIANT HOLDS: operand reads scale with the plan ({per_layer:.1} per \
             layer), not with the container"
        );
    }
}
