//! Greedy speculative generation over a block target, such as the remote
//! [`RoutedBlockSession`].
//!
//! The policy is shared ([`crate::speculation`]): what to propose, how much
//! of a proposal the target agreed with, where generation ends, and the
//! greedy choice. What this loop adds is the execution order against a
//! target that verifies blocks, commits prefixes and steps single
//! positions. It owns no continuation state: the target keeps its K/V and
//! is told what to keep.
//!
//! The local Metal loop (`vindex3 exec --backend metal-lowered
//! --speculate`) uses the same policy. It is a separate loop because its
//! structure is the Metal decode chain's committed look-ahead, which has no
//! counterpart here.
//!
//! Every generated id is recorded once, in an event ledger, with where it
//! came from and whether it was emitted. [`SpeculativeRun::reconcile`]
//! checks that the emitted events are exactly the generated ids, in order.

use std::collections::HashSet;

use crate::error::InferenceError;
use crate::speculation::{accepted_prefix, greedy_id, truncate_at_stop};

use super::routed_block::RoutedBlockSession;
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;

/// What the loop needs from the model side. Implemented by
/// [`RoutedBlockSession`]; a test double can implement it too.
pub trait BlockTarget {
    /// Run `tokens` from the committed position; every position's logits.
    fn verify(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, InferenceError>;
    /// Keep the first `count` positions of the verified block.
    fn commit(&mut self, count: usize) -> Result<(), InferenceError>;
    /// Run and commit one position; its logits.
    fn step(&mut self, token: u32) -> Result<Vec<f32>, InferenceError>;
    /// The committed position.
    fn position(&self) -> usize;
}

impl<B: PlanBackend> BlockTarget for RoutedBlockSession<'_, B> {
    fn verify(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, InferenceError> {
        RoutedBlockSession::verify(self, tokens)
    }
    fn commit(&mut self, count: usize) -> Result<(), InferenceError> {
        RoutedBlockSession::commit(self, count)
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, InferenceError> {
        RoutedBlockSession::step(self, token)
    }
    fn position(&self) -> usize {
        RoutedBlockSession::position(self)
    }
}

/// Proposes tokens for the target to verify.
///
/// Any closure `FnMut(&[u32], usize, usize) -> Vec<u32>` is a proposer, so
/// [`crate::speculation::propose`] can be passed directly.
pub trait Proposer {
    /// Up to `max` tokens to verify after `ctx`, whose first `source_len`
    /// tokens are the prompt. Returning more than `max` is allowed; the loop
    /// cuts the proposal back.
    fn propose(&mut self, ctx: &[u32], source_len: usize, max: usize) -> Vec<u32>;
}

impl<F: FnMut(&[u32], usize, usize) -> Vec<u32> + ?Sized> Proposer for F {
    fn propose(&mut self, ctx: &[u32], source_len: usize, max: usize) -> Vec<u32> {
        self(ctx, source_len, max)
    }
}

/// Where a generated id came from.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The prompt's own greedy id, from the last prefill position.
    Prefill,
    /// A proposed token the target's greedy choice agreed with.
    AcceptedProposal,
    /// The target's greedy id after a verified block's accepted prefix:
    /// the correction of the first rejected proposal, or the next id after
    /// a fully accepted one.
    TargetCorrection,
    /// The target's greedy id from a single step, when nothing was proposed.
    SingleStep,
}

/// One generated id.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub id: u32,
    pub source: Source,
    /// The verify block it came from, for proposals and corrections.
    pub block: Option<usize>,
    /// False when the id came after a stop id or beyond the requested
    /// count. The loop never produces such an id; the field keeps the
    /// ledger honest if that ever changes.
    pub emitted: bool,
}

/// One verified block.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// The committed position the block started from.
    pub start: usize,
    /// The block's tokens: the last generated id, then the proposal.
    pub width: usize,
    pub proposed: usize,
    pub accepted: usize,
    /// Offset within the proposal of the first rejected token, if any.
    pub rejected_at: Option<usize>,
}

/// What one speculative generation did.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpeculativeRun {
    /// The generated ids: at most the requested count, ending at the first
    /// stop id, which is kept.
    pub ids: Vec<u32>,
    pub events: Vec<Event>,
    pub blocks: Vec<Block>,
    pub single_steps: usize,
    /// Prompt length: what the proposer may copy from.
    pub prompt_len: usize,
    /// The target's committed position when the run began. Normally the
    /// prompt's length, but a session may already hold earlier input (a
    /// system prefix, earlier turns), so positions are judged from here.
    pub start_position: usize,
    /// The target's committed position when the run ended.
    pub final_position: usize,
}

impl SpeculativeRun {
    /// Check the ledger against the output and against what ran.
    ///
    /// - The emitted events, in order, are exactly the generated ids, and
    ///   no emitted event follows an unemitted one.
    /// - The first event is the prefill id, and only it.
    /// - Every block's events are contiguous: its accepted proposals, then
    ///   one target correction. Their count matches the block's record;
    ///   its width is its proposal plus one; its rejection offset is its
    ///   accepted count when it rejected anything.
    /// - Each block started at the position its preceding ids imply, and
    ///   the target ended at `start_position + produced - 1` (the last id
    ///   is produced, never fed).
    /// - A run that generated nothing has no events, blocks or steps, and
    ///   left the target where prefill did.
    pub fn reconcile(&self) -> Result<(), String> {
        if self.events.is_empty() {
            let untouched = self.ids.is_empty()
                && self.blocks.is_empty()
                && self.single_steps == 0
                && self.final_position == self.start_position;
            return if untouched {
                Ok(())
            } else {
                Err("a run with no events generated or moved something".into())
            };
        }
        let emitted: Vec<u32> = self
            .events
            .iter()
            .filter(|e| e.emitted)
            .map(|e| e.id)
            .collect();
        if emitted != self.ids {
            return Err(format!(
                "emitted events {emitted:?} are not the generated ids {:?}",
                self.ids
            ));
        }
        if let Some(index) = self.events.iter().position(|e| !e.emitted) {
            if self.events[index..].iter().any(|e| e.emitted) {
                return Err(format!(
                    "an emitted event follows an unemitted one at {index}"
                ));
            }
        }
        if self.events[0].source != Source::Prefill
            || self.events[1..].iter().any(|e| e.source == Source::Prefill)
        {
            return Err("the prefill id is not the first event, once".into());
        }
        let singles = self
            .events
            .iter()
            .filter(|e| e.source == Source::SingleStep)
            .count();
        if singles != self.single_steps {
            return Err(format!(
                "{singles} single-step events, {} single steps",
                self.single_steps
            ));
        }
        for event in &self.events {
            let in_block = matches!(
                event.source,
                Source::AcceptedProposal | Source::TargetCorrection
            );
            if in_block != event.block.is_some() {
                return Err(format!("event {event:?} is attributed to the wrong place"));
            }
        }
        for (index, block) in self.blocks.iter().enumerate() {
            let at: Vec<usize> = (0..self.events.len())
                .filter(|&i| self.events[i].block == Some(index))
                .collect();
            let Some(&first) = at.first() else {
                return Err(format!("block {index} produced no ids"));
            };
            if at.windows(2).any(|w| w[1] != w[0] + 1) {
                return Err(format!("block {index}'s ids are not contiguous"));
            }
            let sources: Vec<Source> = at.iter().map(|&i| self.events[i].source).collect();
            let (correction, proposals) = sources.split_last().expect("non-empty");
            if *correction != Source::TargetCorrection
                || proposals.iter().any(|s| *s != Source::AcceptedProposal)
                || proposals.len() != block.accepted
            {
                return Err(format!(
                    "block {index}: {} accepted, events {sources:?}",
                    block.accepted
                ));
            }
            let rejected = (block.accepted < block.proposed).then_some(block.accepted);
            if block.width != block.proposed + 1
                || block.accepted > block.proposed
                || block.rejected_at != rejected
            {
                return Err(format!("block {index} is inconsistent: {block:?}"));
            }
            if block.start + 1 != self.start_position + first {
                return Err(format!(
                    "block {index} started at {}, {first} ids after the run began at {}",
                    block.start, self.start_position
                ));
            }
        }
        if self.final_position + 1 != self.start_position + self.events.len() {
            return Err(format!(
                "the target ended at {}, {} ids after the run began at {}",
                self.final_position,
                self.events.len(),
                self.start_position
            ));
        }
        Ok(())
    }
}

/// Feed the prompt to `target` in blocks of at most `width` and return the
/// prompt's greedy id: the first generated id.
pub fn prefill<T: BlockTarget + ?Sized>(
    target: &mut T,
    prompt: &[u32],
    width: usize,
) -> Result<u32, InferenceError> {
    if prompt.is_empty() || width == 0 {
        return Err(InferenceError::Parse(
            "prefill needs a prompt and a block width of at least one".into(),
        ));
    }
    let mut last = None;
    for chunk in prompt.chunks(width) {
        let rows = target.verify(chunk)?;
        target.commit(chunk.len())?;
        last = rows.into_iter().last();
    }
    let logits = last.ok_or_else(|| InferenceError::Parse("prefill produced no logits".into()))?;
    greedy(&logits)
}

/// Record one generated id: in the output, the proposer's context and the ledger.
fn push(
    run: &mut SpeculativeRun,
    ctx: &mut Vec<u32>,
    id: u32,
    source: Source,
    block: Option<usize>,
) {
    run.ids.push(id);
    ctx.push(id);
    run.events.push(Event {
        id,
        source,
        block,
        emitted: true,
    });
}

fn greedy(logits: &[f32]) -> Result<u32, InferenceError> {
    greedy_id(logits).ok_or_else(|| {
        InferenceError::Parse("no greedy id: the logits were empty, NaN, +inf or all -inf".into())
    })
}

/// Greedy generation of at most `n` ids after `first` (the prompt's greedy
/// id, already produced by prefill, not yet fed), verifying proposals of up
/// to `width - 1` tokens in blocks of at most `width`.
///
/// The ids are exactly sequential greedy decoding's: a block keeps the
/// prefix on which the target's greedy ids agree with the proposal, plus
/// the target's own id after it, and the target commits those positions.
///
/// No position is verified that the output cannot use. A proposal is cut
/// to the ids still wanted less one (the block's own correction) and cut
/// before its first stop id, so nothing past a stop is ever verified or
/// committed; if the target agrees, its correction supplies the stop.
///
/// **On error** the partial run is not returned, and the target may be
/// left with a verified block pending or a position committed: discard it
/// and start a fresh one.
pub fn greedy_speculate<T: BlockTarget + ?Sized>(
    target: &mut T,
    prompt: &[u32],
    first: u32,
    n: usize,
    width: usize,
    stop: &HashSet<u32>,
    proposer: &mut dyn Proposer,
) -> Result<SpeculativeRun, InferenceError> {
    if width < 2 {
        return Err(InferenceError::Parse(
            "speculation needs a block width of at least two".into(),
        ));
    }
    let mut run = SpeculativeRun {
        prompt_len: prompt.len(),
        start_position: target.position(),
        final_position: target.position(),
        ..Default::default()
    };
    if n == 0 {
        return Ok(run);
    }
    let mut ctx: Vec<u32> = prompt.to_vec();
    push(&mut run, &mut ctx, first, Source::Prefill, None);
    while run.ids.len() < n && !stop.contains(run.ids.last().expect("seeded")) {
        let last = *run.ids.last().expect("seeded");
        // A block yields its accepted proposals plus the target's own next
        // id, so ask for at most one fewer than are still wanted. With
        // nothing left to propose, the proposer is not asked.
        let budget = (width - 1).min(n - run.ids.len() - 1);
        let mut guess = if budget == 0 {
            Vec::new()
        } else {
            proposer.propose(&ctx, prompt.len(), budget)
        };
        // A proposer that returns more than it was asked for is cut back,
        // and a proposal never runs through a stop id.
        guess.truncate(budget);
        if let Some(at) = guess.iter().position(|token| stop.contains(token)) {
            guess.truncate(at);
        }
        if guess.is_empty() {
            let id = greedy(&target.step(last)?)?;
            run.single_steps += 1;
            push(&mut run, &mut ctx, id, Source::SingleStep, None);
            continue;
        }
        let start = target.position();
        let mut block = Vec::with_capacity(guess.len() + 1);
        block.push(last);
        block.extend(&guess);
        let rows = target.verify(&block)?;
        let ids = rows
            .iter()
            .map(|row| greedy(row))
            .collect::<Result<Vec<u32>, _>>()?;
        let accepted = accepted_prefix(&guess, &ids);
        target.commit(accepted + 1)?;
        let index = run.blocks.len();
        run.blocks.push(Block {
            start,
            width: block.len(),
            proposed: guess.len(),
            accepted,
            rejected_at: (accepted < guess.len()).then_some(accepted),
        });
        for &id in &ids[..accepted] {
            push(
                &mut run,
                &mut ctx,
                id,
                Source::AcceptedProposal,
                Some(index),
            );
        }
        push(
            &mut run,
            &mut ctx,
            ids[accepted],
            Source::TargetCorrection,
            Some(index),
        );
    }
    truncate_at_stop(&mut run.ids, stop, n);
    for event in run.events.iter_mut().skip(run.ids.len()) {
        event.emitted = false;
    }
    run.final_position = target.position();
    Ok(run)
}
