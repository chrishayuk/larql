//! Records the balance trajectory and measures neighbours at every step.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use larql_lql::balance_trace::{
    BalanceDecision, BalanceEnd, BalanceObserver, BalanceStep, BalanceTermination, StepContext,
};
use serde_json::{json, Value};

/// A prompt's baseline top-1, measured before the install.
pub type Baseline = HashMap<String, (u32, f64)>;

pub struct Trajectory {
    pub events: Arc<Mutex<Vec<Value>>>,
    pub baseline: Arc<Baseline>,
    pub target_id: u32,
    /// Prompts measured at every step, in addition to the canonical one.
    pub watched: Vec<String>,
}

fn decision(d: &BalanceDecision) -> Value {
    match d {
        BalanceDecision::InBand => json!({"kind": "in_band", "accepted": true}),
        BalanceDecision::Amplify { factor } => {
            json!({"kind": "amplify", "factor": factor, "accepted": false})
        }
        BalanceDecision::Shrink { factor } => {
            json!({"kind": "shrink", "factor": factor, "accepted": false})
        }
        BalanceDecision::Saturated => json!({"kind": "saturated", "accepted": false}),
    }
}

impl Trajectory {
    fn measure(&self, context: &StepContext<'_>, prompt: &str) -> Value {
        match context.top_k(prompt, crate::consts::TOP_K as usize) {
            Ok(top) => {
                let (id, p) = top[0];
                let base = self.baseline.get(prompt);
                json!({
                    "prompt": prompt,
                    "top1_id": id,
                    "top1_p": p,
                    "top1_text": context.decode(id),
                    "p_target_first_token": top.iter().find(|(i, _)| *i == self.target_id).map_or(0.0, |(_, p)| *p as f64),
                    "top1_changed_from_baseline": base.map(|(b, _)| *b != id),
                })
            }
            Err(e) => json!({"prompt": prompt, "error": e.to_string()}),
        }
    }
}

impl BalanceObserver for Trajectory {
    fn step(&mut self, step: &BalanceStep, context: &StepContext<'_>) {
        let canonical = self.measure(context, context.canonical_prompt());
        let watched: Vec<Value> = self
            .watched
            .iter()
            .map(|p| self.measure(context, p))
            .collect();
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(json!({
                "event": "step",
                "iteration": step.iteration,
                "down_norm": step.down_norm,
                "controller_target_prob": step.controller_target_prob,
                "decision": decision(&step.decision),
                "canonical": canonical,
                "watched": watched,
            }));
    }

    fn end(&mut self, end: &BalanceEnd, _context: &StepContext<'_>) {
        let termination = match end.termination {
            BalanceTermination::InBand => "in_band",
            BalanceTermination::Saturated => "saturated",
            BalanceTermination::IterationsExhausted => "iterations_exhausted",
        };
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(json!({
                "event": "end",
                "iterations": end.iterations,
                "termination": termination,
                "rolled_back": end.rolled_back,
                "final_down_norm": end.final_down_norm,
            }));
    }
}
