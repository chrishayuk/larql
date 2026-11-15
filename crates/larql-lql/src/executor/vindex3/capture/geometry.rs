//! What a capture can return, read from the bound program's plan before anything runs: how many layers there
//! are, which token ids a continuation may name, and the most values one position of a site can hold. A caller
//! that must bound a capture's size (the session service) asks here instead of running it.

use crate::ast::CaptureSite;
use crate::error::LqlError;
use crate::executor::{Backend, Session};

/// The plan's bounds on one capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureGeometry {
    /// The program's layers: a capture's layers are below this.
    pub plan_layers: usize,
    /// The embedding's vocabulary, when the plan has an embedding: a continuation's ids are below it.
    pub vocab: Option<usize>,
    /// The most values the site holds at one position. `None` where the plan does not bound it here (a
    /// hyper-connection residual, a routed FFN's down projection input).
    pub width: Option<usize>,
}

impl Session {
    /// The plan's bounds on a capture of `site` at `layers`.
    pub fn capture_geometry(
        &self,
        site: CaptureSite,
        layers: &[usize],
    ) -> Result<CaptureGeometry, LqlError> {
        let runtime = match &self.backend {
            Backend::Vindex3 { runtime, .. } => runtime,
            Backend::None => return Err(LqlError::NoBackend),
            _ => return Err(LqlError::Execution(
                "a capture's geometry is a VINDEX3 program's, and this session is not bound to one"
                    .into(),
            )),
        };
        let plan = runtime.plan();
        let vocab = plan.embedding.as_ref().map(|e| e.vocab_size);
        let hidden = plan
            .embedding
            .as_ref()
            .and_then(|e| e.table.shape.last().copied());
        let widened = plan.hyper_connection_head.is_some();
        let width = match site {
            CaptureSite::Logits => {
                let head = plan
                    .output
                    .as_ref()
                    .and_then(|o| o.projection.shape.first().copied());
                head.into_iter().chain(vocab).max()
            }
            CaptureSite::FfnDownInput => layers
                .iter()
                .map(|&l| {
                    plan.layers
                        .get(l)
                        .and_then(|layer| layer.ffn.as_ref())
                        .and_then(|ffn| ffn.dense())
                        .map(|dense| dense.intermediate_size)
                })
                .collect::<Option<Vec<_>>>()
                .and_then(|sizes| sizes.into_iter().max()),
            _ if widened => None,
            _ => hidden,
        };
        Ok(CaptureGeometry {
            plan_layers: plan.layers.len(),
            vocab,
            width,
        })
    }
}
