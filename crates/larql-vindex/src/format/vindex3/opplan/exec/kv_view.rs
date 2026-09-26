//! The logical continuation view (CONTINUATION-VIEW-1, V2).
//!
//! A kernel reads earlier positions' K and V rows through a [`KvView`]: a
//! logical range `[base, end)` of ABSOLUTE positions over a backing the
//! executor does not see. `end` is the number of rows ever appended — a
//! logical count, never a physical one — and `base` is the first position
//! still held. Row-backed, contiguous and retained-range storage all answer
//! through the same view without being named in it.
//!
//! A read outside `[base, end)` is refused by name ([`ViewRefusal`]).
//! [`AttentionStepCall::new`](super::backend::AttentionStepCall::new)
//! checks the plan's required range against the view before any backend
//! runs, so a provider that dropped a row a step needs is refused there,
//! never discovered by an index panic inside a kernel.

use std::fmt;
use std::ops::Range;

/// Row storage a view reads, by ABSOLUTE position. Called only for
/// positions inside the owning view's `[base, end)`.
pub trait KvRows {
    fn row(&self, position: usize) -> &[f32];
}

/// Where a view's rows live. Rows held one allocation per position are
/// read directly (no dispatch — today's providers); any other layout is
/// read through [`KvRows`], so a new backing needs no change here.
#[derive(Clone, Copy)]
enum Backing<'a> {
    /// `rows[i]` is position `base + i`.
    Rows(&'a [Vec<f32>]),
    Dyn(&'a (dyn KvRows + Sync)),
}

impl<'a> Backing<'a> {
    fn row(self, base: usize, position: usize) -> &'a [f32] {
        match self {
            Self::Rows(rows) => &rows[position - base],
            Self::Dyn(rows) => rows.row(position),
        }
    }
}

/// The K and V rows a step may read, over positions `[base, end)`.
#[derive(Clone, Copy)]
pub struct KvView<'a> {
    base: usize,
    end: usize,
    keys: Backing<'a>,
    values: Backing<'a>,
}

impl fmt::Debug for KvView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KvView[{}..{})", self.base, self.end)
    }
}

/// Nothing held: the view of a sequence's first position.
static NO_ROWS: [Vec<f32>; 0] = [];

impl<'a> KvView<'a> {
    /// A view over `[base, end)` read through [`KvRows`]. Refuses
    /// `base > end`.
    pub fn new(
        base: usize,
        end: usize,
        keys: &'a (dyn KvRows + Sync),
        values: &'a (dyn KvRows + Sync),
    ) -> Result<Self, ViewRefusal> {
        Self::checked(base, end, Backing::Dyn(keys), Backing::Dyn(values))
    }

    /// Rows held one allocation per position, the first at `base`: row `i`
    /// of each slice is position `base + i`.
    pub fn rows_from(
        base: usize,
        keys: &'a [Vec<f32>],
        values: &'a [Vec<f32>],
    ) -> Result<Self, ViewRefusal> {
        assert_eq!(
            keys.len(),
            values.len(),
            "a provider lent {} K rows and {} V rows",
            keys.len(),
            values.len()
        );
        Self::checked(
            base,
            base + keys.len(),
            Backing::Rows(keys),
            Backing::Rows(values),
        )
    }

    /// Every row from position 0, as a provider's `keys` / `values` lend
    /// them today.
    pub fn over_rows(keys: &'a [Vec<f32>], values: &'a [Vec<f32>]) -> Self {
        Self::rows_from(0, keys, values).expect("base 0 never exceeds end")
    }

    /// No rows held: position 0 of a fresh sequence.
    pub fn empty() -> KvView<'static> {
        KvView::over_rows(&NO_ROWS, &NO_ROWS)
    }

    fn checked(
        base: usize,
        end: usize,
        keys: Backing<'a>,
        values: Backing<'a>,
    ) -> Result<Self, ViewRefusal> {
        if base > end {
            return Err(ViewRefusal {
                needed: base..base,
                base,
                end,
            });
        }
        Ok(Self {
            base,
            end,
            keys,
            values,
        })
    }

    /// The first position still held.
    pub fn base(&self) -> usize {
        self.base
    }

    /// One past the last position appended: a logical count of rows ever
    /// appended, not of rows physically held.
    pub fn end(&self) -> usize {
        self.end
    }

    /// Whether every position in `needed` is held.
    pub fn covers(&self, needed: Range<usize>) -> Result<(), ViewRefusal> {
        let empty = needed.start >= needed.end;
        if empty || (self.base <= needed.start && needed.end <= self.end) {
            Ok(())
        } else {
            Err(ViewRefusal {
                needed,
                base: self.base,
                end: self.end,
            })
        }
    }

    /// K row at absolute `position`. Callers check [`covers`](Self::covers)
    /// first (AttentionStepCall::new always does), so an out-of-range read
    /// is an executor bug, not a retention one. Named in debug builds; in
    /// release the backing's own bounds check still panics, without the
    /// name — this is the hottest call in decode, and the per-read check
    /// measured at ≈2.7% of qwen3-0.6b decode (continuation-view-1-notes).
    pub fn key(&self, position: usize) -> &'a [f32] {
        self.check(position);
        self.keys.row(self.base, position)
    }

    /// V row at absolute `position`; see [`key`](Self::key).
    pub fn value(&self, position: usize) -> &'a [f32] {
        self.check(position);
        self.values.row(self.base, position)
    }

    fn check(&self, position: usize) {
        debug_assert!(
            self.base <= position && position < self.end,
            "executor bug: read position {position} outside the checked view [{}, {})",
            self.base,
            self.end
        );
    }
}

/// A step needed positions the view does not hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewRefusal {
    /// The positions the plan requires.
    pub needed: Range<usize>,
    /// The first position the view holds.
    pub base: usize,
    /// One past the last position the view holds.
    pub end: usize,
}

impl fmt::Display for ViewRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "continuation view holds positions [{}, {}) but the plan requires [{}, {}); \
             a provider may drop only rows before the plan's required range",
            self.base, self.end, self.needed.start, self.needed.end
        )
    }
}

impl std::error::Error for ViewRefusal {}

impl From<ViewRefusal> for crate::error::VindexError {
    fn from(value: ViewRefusal) -> Self {
        Self::Parse(value.to_string())
    }
}
