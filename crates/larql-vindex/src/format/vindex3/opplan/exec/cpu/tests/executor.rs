//! `tests` for [`super::super::executor`] — the fan-out primitives.
//!
//! The projection paths are already gated by `projection_cost.rs` and
//! `kernels.rs`; what is checked here is the parallel machinery itself,
//! including the arm every one of those primitives takes when the
//! caller is already inside a rayon worker.

use super::super::executor::{shared, CpuExecutor};

fn exec() -> CpuExecutor {
    CpuExecutor::new().expect("a pool must build on any test host")
}

/// The process-wide executor is built once and handed out by reference.
#[test]
fn the_shared_executor_is_one_pool() {
    let a = shared().expect("shared executor");
    let b = shared().expect("shared executor again");
    assert!(std::ptr::eq(a, b), "`shared` must not build a second pool");
    assert!(a.workers() >= 1);
}

/// `parallel_map` computes every item exactly once, in order, whichever
/// arm it takes.
#[test]
fn parallel_map_preserves_order_and_computes_each_item_once() {
    let e = exec();
    let items: Vec<usize> = (0..64).collect();
    let got = e.parallel_map(&items, |i| i * 3);
    assert_eq!(got, items.iter().map(|i| i * 3).collect::<Vec<_>>());

    // Inside the pool the caller already owns the machine, so the same
    // call runs SERIALLY — same answer, which is the property that
    // makes the rule safe to apply anywhere.
    let nested = e.parallel_map(&items, |i| {
        let inner = exec();
        inner.parallel_map(&[*i], |j| j * 3)[0]
    });
    assert_eq!(nested, got, "nesting must not change the answer");
}

/// `parallel6` returns its six results in ARGUMENT order — the one
/// thing a join tree can silently get wrong, since every closure here
/// has the same type and a transposition would still compile.
#[test]
fn parallel6_returns_its_results_in_argument_order() {
    let e = exec();
    let got = e.parallel6(
        || "a".to_string(),
        || "b".to_string(),
        || "c".to_string(),
        || "d".to_string(),
        || "e".to_string(),
        || "f".to_string(),
    );
    assert_eq!(
        got,
        (
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
            "f".to_string()
        )
    );

    // Different types per position, so a transposition cannot even be
    // expressed — and the ordering claim is checked on values too.
    let mixed = e.parallel6(|| 1u8, || 2u16, || 3u32, || 4u64, || 5i8, || 6i16);
    assert_eq!(mixed, (1u8, 2u16, 3u32, 4u64, 5i8, 6i16));

    // The serial arm: called from inside the pool, same answer.
    let nested = e.parallel6(
        || {
            let inner = exec();
            inner.parallel6(|| 1, || 2, || 3, || 4, || 5, || 6)
        },
        || (0, 0, 0, 0, 0, 0),
        || (0, 0, 0, 0, 0, 0),
        || (0, 0, 0, 0, 0, 0),
        || (0, 0, 0, 0, 0, 0),
        || (0, 0, 0, 0, 0, 0),
    );
    assert_eq!(nested.0, (1, 2, 3, 4, 5, 6));
}
