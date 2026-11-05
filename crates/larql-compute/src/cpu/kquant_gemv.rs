//! The CPU's K-quant gemv — the stored block, multiplied where it lies.
//!
//! **Not [`crate::pipeline::quant_format::QuantFormat::Q8_0`].** That
//! name already exists in this crate and means a different thing: int8
//! codes with an EXTERNAL per-block f32 scale array, block 32. The
//! formats here are ggml's, whose scale is INSIDE the block — Q8_0 is
//! one f16 then 32 codes in 34 bytes. Two layouts sharing a name is how
//! a reader ends up passing one to the other's kernel, so this module
//! says `ggml` in its types and keeps its distance.
//!
//! Until this existed, a K-quant operand had exactly one execution
//! route: decode the whole matrix to f32 and run an f32 gemv over the
//! result. That is correct and it is what PARETO-1's rung A was measured
//! with, but it prices a Q8_0 model at **four bytes per weight**. On
//! Qwen3.8-27B the profiler put 97.3% of a decode token in `Projection`,
//! moving 97.4 GB of widened f32 per token where the artifact itself
//! holds the same information in 24.4 GB.
//!
//! The arithmetic here is not a second opinion about the format. It is
//! the association [`larql_models::quant::ggml::dequantize_q8_0`]
//! documents — `(q as i8 as f32) * f16(d)` — applied one block at a time
//! instead of into a whole decoded matrix, and multiplied into the
//! accumulator in element order. That makes it match
//! dequantise-then-multiply **exactly** rather than approximately, and
//! the tests hold it to that oracle.
//!
//! # Why exactness is available here at all
//!
//! `d` is an f16 and `q` an int8, so `d * q` needs at most 11 + 8 bits
//! of significand and is therefore EXACT in f32. Folding the scale into
//! each weight before the multiply — rather than hoisting it out of the
//! block's dot product, which would be cheaper — is what lets this
//! kernel be bit-for-bit with the decoder instead of merely close. A
//! hoisted scale is a different program, and on a path whose whole
//! purpose is to leave kernel quality out of a behavioural curve, "a
//! different program" is the thing to avoid.
//!
//! # Threading
//!
//! The kernel threads itself, as the Q4_K and Q6_K kernels beside it do:
//! rows are independent, so a row-parallel schedule changes which thread
//! runs a row and nothing about what that row computes — every row's
//! accumulation is the serial loop in `row_dot`, in the same order.
//! A caller that threads on top (LARQL's plan executor) declares this
//! kernel library-owned and calls it once.
//!
//! The threads are rayon's and their number is the current pool's, but a
//! fork-join would cost more than the work: at this kernel's speed a whole
//! gemv is tens of microseconds, about what it takes rayon to wake a
//! blocked caller. So the caller works instead of blocking, and helpers
//! claim chunks from a published job, and give the worker back to rayon as
//! soon as no further job is waiting (see [`LINGER`]). See [`Job`] and
//! [`helper`].
//!
//! # Why the lanes hold rows
//!
//! Addition order within a row is fixed, so a row's additions cannot run
//! in parallel with each other; alone, a row is bound by the latency of
//! its add chain. Different rows have no such constraint, so the NEON path
//! gives each lane its own row and its own accumulator. Every lane still
//! performs exactly its row's operations in its row's order.

use larql_models::quant::half::f16_to_f32;
use std::any::Any;
use std::cell::Cell;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicPtr, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Elements described by one Q8_0 block.
pub const Q8_0_BLOCK_ELEMS: usize = 32;

/// Bytes one Q8_0 block occupies: an f16 scale then 32 signed codes.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Test-only fault injection: a panic in a chunk of the job whose row count is `FAIL_N`, either in a helper (waiting
/// for one to arrive, so the helper is the one that panics) or in the calling thread.
#[cfg(test)]
pub(crate) static FAIL_N: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static FAIL_IN_HELPER: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static HELPER_PANICKED: AtomicBool = AtomicBool::new(false);
/// The thread that called `q8_0_gemv` for the injected job. A thread-local flag cannot say: a caller can itself be
/// running inside a lingering helper's `yield_now`.
#[cfg(test)]
static FAIL_CALLER: Mutex<Option<std::thread::ThreadId>> = Mutex::new(None);

/// Two workers panic together, the second with a payload whose destructor panics.
#[cfg(test)]
pub(crate) static FAIL_TWO_PANICS: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static SECOND_ARRIVED: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
pub(crate) struct PanicsOnDrop;
#[cfg(test)]
impl Drop for PanicsOnDrop {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            panic!("payload destructor panicked");
        }
    }
}

#[cfg(test)]
fn failpoint(n: usize) {
    if n != FAIL_N.load(Ordering::SeqCst) {
        return;
    }
    let helper =
        *FAIL_CALLER.lock().unwrap_or_else(|e| e.into_inner()) != Some(std::thread::current().id());
    if FAIL_TWO_PANICS.load(Ordering::SeqCst) {
        if helper {
            SECOND_ARRIVED.store(true, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(150)); // the caller's panic is recorded first
            std::panic::panic_any(PanicsOnDrop);
        }
        let t = Instant::now();
        while !SECOND_ARRIVED.load(Ordering::SeqCst) && t.elapsed() < Duration::from_secs(5) {
            std::hint::spin_loop();
        }
        panic!("injected caller panic");
    }
    if FAIL_IN_HELPER.load(Ordering::SeqCst) {
        if helper {
            HELPER_PANICKED.store(true, Ordering::SeqCst);
            panic!("injected helper panic");
        }
        let t = Instant::now();
        while !HELPER_PANICKED.load(Ordering::SeqCst) && t.elapsed() < Duration::from_secs(5) {
            std::hint::spin_loop();
        }
    } else if !helper {
        panic!("injected caller panic");
    }
}

/// Rows the kernel prefers to process together; chunks are multiples of it.
const GROUP_ROWS: usize = 16;

/// Elements per parallel work unit. Large enough that threads rarely meet
/// on the claim counter, small enough that a slow core holding the last
/// chunk does not hold up the result for long.
const CHUNK_ELEMS: usize = 32 * 1024;

/// Wake-ups one thread issues itself when bringing up helpers.
const FANOUT: usize = 2;

/// How long the caller spins on the last chunks before it starts yielding.
const SPINS_BEFORE_YIELD: u32 = 1 << 14;

/// How long a helper waits for another job before returning to rayon.
///
/// Zero: a helper checks once (after [`POLLS_PER_CHECK`] polls) and leaves if no
/// new job has arrived. Staying resident for longer (25 us was tried) makes
/// back-to-back gemvs faster on one stream, but a resident helper is a rayon
/// worker that is not scheduling: with decode streams running, unrelated work
/// in the same rayon pool then stalls, with fork-joins finishing hundreds of
/// milliseconds late, and almost none of it completes. Measured against
/// that, zero keeps most of the speedup and leaves the pool usable.
const LINGER: Duration = Duration::from_micros(0);

/// Polls of the job counter between looks at the clock and at rayon's queues.
const POLLS_PER_CHECK: u32 = 256;

/// Rows per chunk for a `k`-column matrix.
fn chunk_rows(k: usize) -> usize {
    (CHUNK_ELEMS / (k * GROUP_ROWS)).max(1) * GROUP_ROWS
}

/// `out[n] = W[n, k] · x[k]`, with `W` read straight from the stored
/// Q8_0 pack.
///
/// `None` — never a wrong answer — when the geometry does not describe
/// this pack: `k` off the block grid, or the stream the wrong length for
/// `[n, k]`. Silently accepting a short stream would read a neighbouring
/// row's codes as this row's and return a plausible vector.
pub fn q8_0_gemv(blocks: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
    if k == 0 || !k.is_multiple_of(Q8_0_BLOCK_ELEMS) || x.len() != k {
        return None;
    }
    let per_row = k / Q8_0_BLOCK_ELEMS;
    let expected = n.checked_mul(per_row)?.checked_mul(Q8_0_BLOCK_BYTES)?;
    if blocks.len() != expected {
        return None;
    }

    let mut out = vec![0.0f32; n];
    let threads = rayon::current_num_threads();
    let chunk_rows = chunk_rows(k);
    if threads <= 1 || n <= chunk_rows {
        run_rows(blocks, x, 0, per_row, &mut out);
        return Some(out);
    }

    let job = Job {
        blocks: blocks.as_ptr(),
        blocks_len: blocks.len(),
        x: x.as_ptr(),
        out: out.as_mut_ptr(),
        n,
        per_row,
        chunk_rows,
        chunks: n.div_ceil(chunk_rows),
        seats: AtomicIsize::new(threads as isize - 1),
        panicked: AtomicBool::new(false),
        payload: Mutex::new(None),
        next: Padded(AtomicUsize::new(0)),
        done: Padded(AtomicUsize::new(0)),
    };
    #[cfg(test)]
    if n == FAIL_N.load(Ordering::SeqCst) {
        *FAIL_CALLER.lock().unwrap_or_else(|e| e.into_inner()) = Some(std::thread::current().id());
    }
    let Some(published) = Published::new(&job) else {
        // Another gemv holds the slot, so fork-join the ordinary way.
        use rayon::prelude::*;
        out.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk, slots)| {
                run_rows(blocks, x, chunk * chunk_rows, per_row, slots);
            });
        return Some(out);
    };
    let lingering = LINGERING.0.load(Ordering::SeqCst);
    launch((threads - 1).saturating_sub(lingering));
    job.work();
    let mut spins = 0u32;
    while job.done.0.load(Ordering::Acquire) != job.chunks {
        spins = spins.saturating_add(1);
        if spins < SPINS_BEFORE_YIELD {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
    drop(published);
    if job.panicked.load(Ordering::Acquire) {
        // Every helper has left the job, and `out` is dropped unread: no caller sees a result with uncomputed rows.
        let payload = job.payload.lock().unwrap_or_else(|e| e.into_inner()).take();
        drop(out);
        resume_unwind(payload.unwrap_or_else(|| Box::new("q8_0_gemv: a worker panicked")));
    }
    Some(out)
}

/// The job helpers should join, or null. Only a [`Published`] writes it.
static CURRENT: Padded<AtomicPtr<Job>> = Padded(AtomicPtr::new(std::ptr::null_mut()));
/// Bumped per publication, so helpers poll a counter and not the job.
static GENERATION: Padded<AtomicUsize> = Padded(AtomicUsize::new(0));
/// Helpers currently waiting for a job, which need no spawn to reach.
static LINGERING: Padded<AtomicUsize> = Padded(AtomicUsize::new(0));
/// Helpers that may be holding a pointer read from [`CURRENT`].
static ACTIVE: Padded<AtomicUsize> = Padded(AtomicUsize::new(0));

thread_local! {
    /// Set while this thread is in [`helper`], whose rayon yields can run
    /// another queued helper on the same stack.
    static IN_HELPER: Cell<bool> = const { Cell::new(false) };
}

/// One gemv's shared state, on the caller's stack. Chunks are claimed by
/// counter, so whichever threads turn up do the work and the caller never
/// sleeps on a latch.
///
/// The claim counter and the completion counter each have their own cache
/// line: everything else here is read on every claim, and would otherwise
/// be invalidated by every claim.
struct Job {
    blocks: *const u8,
    blocks_len: usize,
    x: *const f32,
    out: *mut f32,
    n: usize,
    per_row: usize,
    chunk_rows: usize,
    chunks: usize,
    /// Helpers still allowed in: the caller's pool size is the limit, even
    /// if a thread lingering from another pool is the one that notices.
    seats: AtomicIsize,
    /// Set when a chunk panicked. Every chunk is still counted as finished, so the caller is never left waiting; it then
    /// withdraws the job, and re-raises the panic instead of returning rows that were not computed.
    panicked: AtomicBool,
    payload: Mutex<Option<Box<dyn Any + Send>>>,
    next: Padded<AtomicUsize>,
    done: Padded<AtomicUsize>,
}

/// Apple silicon's cache line.
#[repr(align(128))]
struct Padded<T>(T);

impl Job {
    /// Keep the first panic's payload for the caller to re-raise. Later payloads are discarded, and discarding one must not
    /// unwind: its destructor may itself panic, and unwinding out of [`Job::work`] would lose this thread's finished-chunk
    /// count and leave the caller waiting for chunks that are never counted.
    fn record_panic(&self, payload: Box<dyn Any + Send>) {
        let mut slot = self.payload.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(payload);
            return;
        }
        drop(slot);
        if let Err(also) = catch_unwind(AssertUnwindSafe(move || drop(payload))) {
            std::mem::forget(also); // a payload that panics while dropped is leaked, not re-raised
        }
    }

    /// Claim and run chunks until none are left.
    fn work(&self) {
        let mut finished = 0;
        loop {
            let chunk = self.next.0.fetch_add(1, Ordering::Relaxed);
            if chunk >= self.chunks {
                if finished > 0 {
                    self.done.0.fetch_add(finished, Ordering::Release);
                }
                return;
            }
            let first = chunk * self.chunk_rows;
            let len = self.chunk_rows.min(self.n - first);
            // SAFETY: a thread reaches here either as the caller, whose
            // borrows outlive this call, or as a helper counted in `ACTIVE`,
            // which `Published::drop` waits out before those borrows end.
            // `fetch_add` hands each chunk to one thread, so the `out` range
            // is exclusive, and the caller does not touch `out` until `done`
            // reaches `chunks`. The lengths are the ones `q8_0_gemv` validated.
            // A chunk is counted as finished whether or not it panicked, so `done` always reaches `chunks`. After a panic
            // the remaining chunks are claimed and counted without running.
            if !self.panicked.load(Ordering::Relaxed) {
                let ran = catch_unwind(AssertUnwindSafe(|| {
                    #[cfg(test)]
                    failpoint(self.n);
                    let (blocks, x, slots) = unsafe {
                        (
                            std::slice::from_raw_parts(self.blocks, self.blocks_len),
                            std::slice::from_raw_parts(self.x, self.per_row * Q8_0_BLOCK_ELEMS),
                            std::slice::from_raw_parts_mut(self.out.add(first), len),
                        )
                    };
                    run_rows(blocks, x, first, self.per_row, slots);
                }));
                if let Err(payload) = ran {
                    self.record_panic(payload);
                    self.panicked.store(true, Ordering::Release);
                }
            }
            finished += 1;
        }
    }
}

/// A job visible to helpers for as long as this lives.
struct Published;

impl Published {
    /// `None` when another gemv already holds the slot.
    fn new(job: &Job) -> Option<Self> {
        let job = std::ptr::from_ref(job).cast_mut();
        CURRENT
            .0
            .compare_exchange(
                std::ptr::null_mut(),
                job,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .ok()?;
        GENERATION.0.fetch_add(1, Ordering::SeqCst);
        Some(Self)
    }
}

impl Drop for Published {
    /// Withdraw the job and wait out any helper that might have read it. With
    /// [`join_current`] this is a Dekker pair over sequentially consistent
    /// operations: a helper that saw the job is seen here in `ACTIVE`.
    fn drop(&mut self) {
        CURRENT.0.store(std::ptr::null_mut(), Ordering::SeqCst);
        while ACTIVE.0.load(Ordering::SeqCst) != 0 {
            std::hint::spin_loop();
        }
    }
}

/// Work on whatever job is published right now, if there is a seat.
fn join_current() {
    /// Releases the `ACTIVE` count however this function ends, so a withdrawing caller can never wait forever on it.
    struct Active;
    impl Drop for Active {
        fn drop(&mut self) {
            ACTIVE.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    ACTIVE.0.fetch_add(1, Ordering::SeqCst);
    let _active = Active;
    let job = CURRENT.0.load(Ordering::SeqCst);
    // SAFETY: non-null means a `Published` holds a live job, and its drop
    // cannot complete until the `ACTIVE` count taken above is released.
    if let Some(job) = unsafe { job.as_ref() } {
        if job.seats.fetch_sub(1, Ordering::Relaxed) > 0 {
            job.work();
        }
    }
}

/// A helper's life: join jobs as they are published, and return to rayon
/// after [`LINGER`] without one. While waiting it also yields to rayon, so
/// other work in the pool is run and not delayed; if there is other work, it
/// stops waiting.
fn helper() {
    if IN_HELPER.with(|h| h.replace(true)) {
        return;
    }
    /// Undoes this helper's registration however it ends.
    struct Registered;
    impl Drop for Registered {
        fn drop(&mut self) {
            LINGERING.0.fetch_sub(1, Ordering::SeqCst);
            IN_HELPER.with(|h| h.set(false));
        }
    }
    LINGERING.0.fetch_add(1, Ordering::SeqCst);
    let _registered = Registered;
    let mut seen = 0;
    let mut waiting_since = Instant::now();
    let mut polls = 0u32;
    loop {
        let generation = GENERATION.0.load(Ordering::Acquire);
        if generation != seen {
            seen = generation;
            join_current();
            waiting_since = Instant::now();
        }
        std::hint::spin_loop();
        polls += 1;
        if polls == POLLS_PER_CHECK {
            polls = 0;
            if waiting_since.elapsed() > LINGER
                || matches!(rayon::yield_now(), Some(rayon::Yield::Executed))
            {
                break;
            }
        }
    }
}

/// Bring up to `helpers` more threads, as a tree: each new thread starts its
/// share of the rest, so no thread pays for every wake-up.
fn launch(helpers: usize) {
    let children = helpers.min(FANOUT);
    let mut rest = helpers - children;
    for c in 0..children {
        let share = rest / (children - c);
        rest -= share;
        rayon::spawn(move || {
            launch(share);
            helper();
        });
    }
}

/// Fill `slots` with the dots of consecutive rows starting at `first`.
/// Whole groups go through the interleaved kernel where there is one;
/// what is left goes through [`row_dot`], the same program a row at a time.
#[inline]
fn run_rows(blocks: &[u8], x: &[f32], first: usize, per_row: usize, slots: &mut [f32]) {
    #[allow(unused_mut)]
    let mut done = 0;
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        let mut groups = slots.chunks_exact_mut(neon::ROWS);
        for group in &mut groups {
            let group: &mut [f32; neon::ROWS] =
                group.try_into().expect("chunks_exact_mut yields ROWS");
            neon::rows(blocks, x, first + done, per_row, group);
            done += neon::ROWS;
        }
    }
    for (i, slot) in slots[done..].iter_mut().enumerate() {
        *slot = row_dot(blocks, x, first + done + i, per_row);
    }
}

/// The NEON path reads each scale as a little-endian half and builds codes in the top byte of a lane, so it is
/// little-endian only; any other target takes [`row_dot`].
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon {
    use super::{Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};
    use core::arch::aarch64::*;
    use core::arch::asm;

    pub(super) const ROWS: usize = super::GROUP_ROWS;
    const GROUPS: usize = ROWS / 4;
    const SPAN: usize = 8;
    const SPANS: usize = Q8_0_BLOCK_ELEMS / SPAN;

    /// Each 32-bit lane takes one byte of a column as its TOP byte, so a
    /// fixed-point conversion with 24 fraction bits yields the code itself.
    /// The two table registers hold rows 0, 1 and rows 2, 3 of a group, eight
    /// columns each; entry `j` picks column `j` of all four rows.
    const TOP_BYTE: [[u8; 16]; SPAN] = {
        let mut t = [[0xff; 16]; SPAN];
        let mut j = 0;
        while j < SPAN {
            let mut l = 0;
            while l < 4 {
                t[j][l * 4 + 3] = (l * SPAN + j) as u8;
                l += 1;
            }
            j += 1;
        }
        t
    };

    /// `ROWS` consecutive rows, one per lane. Each lane's accumulator sees
    /// exactly its row's additions in ascending column order; only the
    /// assignment of rows to lanes is new.
    pub(super) fn rows(
        blocks: &[u8],
        x: &[f32],
        first: usize,
        per_row: usize,
        out: &mut [f32; ROWS],
    ) {
        let stride = per_row * Q8_0_BLOCK_BYTES;
        assert!((first + ROWS) * stride <= blocks.len() && x.len() == per_row * Q8_0_BLOCK_ELEMS);
        // SAFETY: the assert above puts `ROWS` whole rows at `first` inside
        // `blocks` and `per_row` whole blocks in `x`; NEON is baseline on
        // aarch64.
        unsafe {
            rows_inner(
                blocks.as_ptr().add(first * stride),
                x.as_ptr(),
                stride,
                per_row,
                out,
            )
        }
    }

    /// binary16 to binary32 for four lanes in one instruction.
    #[inline(always)]
    fn f16_to_f32_lanes(h: uint16x4_t) -> float32x4_t {
        let out: float32x4_t;
        // SAFETY: a register-to-register conversion on baseline NEON; it
        // touches no memory and no flags.
        unsafe {
            asm!("fcvtl {o:v}.4s, {h:v}.4h", o = lateout(vreg) out, h = in(vreg) h,
                 options(pure, nomem, nostack, preserves_flags));
        }
        out
    }

    /// One column: lane `L` of each group's code register, `(q * d) * x`.
    #[inline(always)]
    fn step<const L: i32>(
        acc: &mut [float32x4_t; GROUPS],
        t: &[uint8x16x2_t; GROUPS],
        tb: uint8x16_t,
        d: &[float32x4_t; GROUPS],
        xv: float32x4_t,
    ) {
        // SAFETY: NEON is baseline on aarch64.
        unsafe {
            for g in 0..GROUPS {
                let code = vreinterpretq_s32_u8(vqtbl2q_u8(t[g], tb));
                let q = vcvtq_n_f32_s32::<24>(code);
                acc[g] = vaddq_f32(acc[g], vmulq_laneq_f32::<L>(vmulq_f32(q, d[g]), xv));
            }
        }
    }

    /// # Safety
    /// `p` addresses `ROWS` rows of `stride` bytes and `x` `per_row` blocks.
    #[inline(never)]
    unsafe fn rows_inner(
        p: *const u8,
        x: *const f32,
        stride: usize,
        per_row: usize,
        out: &mut [f32; ROWS],
    ) {
        // SAFETY: bounds per this function's contract; loads are unaligned.
        unsafe {
            let tb = [
                vld1q_u8(TOP_BYTE[0].as_ptr()),
                vld1q_u8(TOP_BYTE[1].as_ptr()),
                vld1q_u8(TOP_BYTE[2].as_ptr()),
                vld1q_u8(TOP_BYTE[3].as_ptr()),
                vld1q_u8(TOP_BYTE[4].as_ptr()),
                vld1q_u8(TOP_BYTE[5].as_ptr()),
                vld1q_u8(TOP_BYTE[6].as_ptr()),
                vld1q_u8(TOP_BYTE[7].as_ptr()),
            ];
            let mut acc = [vdupq_n_f32(0.0); GROUPS];
            for b in 0..per_row {
                let base = p.add(b * Q8_0_BLOCK_BYTES);
                let xs = x.add(b * Q8_0_BLOCK_ELEMS);
                let mut d = [vdupq_n_f32(0.0); GROUPS];
                for (g, dv) in d.iter_mut().enumerate() {
                    let mut h = [0u16; 4];
                    for (l, hl) in h.iter_mut().enumerate() {
                        let at = base.add((g * 4 + l) * stride) as *const u16;
                        *hl = u16::from_le(at.read_unaligned());
                    }
                    *dv = f16_to_f32_lanes(vld1_u16(h.as_ptr()));
                }
                for span in 0..SPANS {
                    let off = 2 + span * SPAN;
                    let mut t = [uint8x16x2_t(vdupq_n_u8(0), vdupq_n_u8(0)); GROUPS];
                    for (g, tg) in t.iter_mut().enumerate() {
                        let row = |l: usize| vld1_u8(base.add((g * 4 + l) * stride + off));
                        *tg =
                            uint8x16x2_t(vcombine_u8(row(0), row(1)), vcombine_u8(row(2), row(3)));
                    }
                    let xh = xs.add(span * SPAN);
                    let xa = vld1q_f32(xh);
                    let xb = vld1q_f32(xh.add(4));
                    step::<0>(&mut acc, &t, tb[0], &d, xa);
                    step::<1>(&mut acc, &t, tb[1], &d, xa);
                    step::<2>(&mut acc, &t, tb[2], &d, xa);
                    step::<3>(&mut acc, &t, tb[3], &d, xa);
                    step::<0>(&mut acc, &t, tb[4], &d, xb);
                    step::<1>(&mut acc, &t, tb[5], &d, xb);
                    step::<2>(&mut acc, &t, tb[6], &d, xb);
                    step::<3>(&mut acc, &t, tb[7], &d, xb);
                }
            }
            for (g, a) in acc.iter().enumerate() {
                vst1q_f32(out.as_mut_ptr().add(g * 4), *a);
            }
        }
    }
}

/// One row's dot in the decoder's association — `(q * d) * x`, element
/// by element, blocks in order. The whole arithmetic of the kernel is
/// here; [`q8_0_gemv`] only decides which thread runs it.
#[inline]
fn row_dot(blocks: &[u8], x: &[f32], row: usize, per_row: usize) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..per_row {
        let base = (row * per_row + b) * Q8_0_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([blocks[base], blocks[base + 1]]));
        let codes = &blocks[base + 2..base + Q8_0_BLOCK_BYTES];
        let xs = &x[b * Q8_0_BLOCK_ELEMS..][..Q8_0_BLOCK_ELEMS];
        for (j, &c) in codes.iter().enumerate() {
            // `d * q` first, then the activation — the decoder's
            // association, element by element. See the module note on
            // why this is exact and why hoisting `d` would not be.
            acc += ((c as i8 as f32) * d) * xs[j];
        }
    }
    acc
}

#[cfg(test)]
#[path = "kquant_gemv_tests.rs"]
mod tests;
