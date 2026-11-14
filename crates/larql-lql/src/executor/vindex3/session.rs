//! Session statements over a VINDEX3 runtime, and knowledge binding.

use crate::error::LqlError;
use crate::executor::{Backend, Session};
use larql_inference::vindex3::{OpenPolicy, Vindex3Runtime};
use larql_inference::SamplingConfig;
use larql_kv::CanonicalFactory;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::ContinuationFactory;
use larql_vindex::format::vindex3::opplan::exec::lowering::{LoweringIdentity, LoweringRegistry};
use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;
use larql_vindex::tokenizers::Tokenizer;

#[allow(unused_imports)]
use super::*;

impl Session {
    /// `EXPLAIN INFER` on a V3 binding (LQL-2): render the structured
    /// [`ExplainPlan`] — the executable authority that will run, not a
    /// reconstruction. Static: no tokens execute.
    pub(crate) fn exec_v3_explain(&self) -> Result<Vec<String>, LqlError> {
        let Backend::Vindex3 { runtime, .. } = &self.backend else {
            unreachable!("caller matched the backend");
        };
        let explain = larql_inference::vindex3::ExplainPlan::from_runtime(runtime);

        let mut out = Vec::new();
        out.push("MODEL".into());
        out.push(format!("  name: {}", explain.model));
        out.push(format!("  generation: {}", explain.generation));
        out.push(format!("  component: {}", explain.component));
        out.push("  execution: closed".into());
        out.push(String::new());
        out.push("PLAN".into());
        out.push(format!(
            "  embedding  vocab {} scaled {} normed {}",
            explain.embedding.vocab_size, explain.embedding.scaled, explain.embedding.normed,
        ));
        for layer in &explain.layers {
            out.push(format!("  layer {}", layer.layer));
            for op in &layer.ops {
                match op.as_str() {
                    "attention" => {
                        let a = &layer.attention;
                        let num = |v: Option<usize>| v.map_or("-".to_string(), |n| n.to_string());
                        out.push(format!(
                            "    attention  mode {} window {}  q/kv {}/{}  head_dim {}{}{}",
                            a.mode,
                            a.window.map_or("-".into(), |w| w.to_string()),
                            num(a.q_heads),
                            num(a.kv_heads),
                            num(a.head_dim),
                            if a.gated { "  gated" } else { "" },
                            if a.qk_norm { "  qk_norm" } else { "" },
                        ));
                        // A recurrence states the one number a softmax
                        // layer cannot: state that does not grow with the
                        // sequence.
                        if let Some(elements) = a.state_elements {
                            out.push(format!("      state       {elements} elements/layer"));
                        }
                        if a.sinks || a.biased {
                            out.push(format!(
                                "      extras      {}{}",
                                if a.sinks { " sinks" } else { "" },
                                if a.biased { " qkvo_bias" } else { "" },
                            ));
                        }
                        for operand in &a.operands {
                            out.push(format!(
                                "      {:12} {}::{} @{}",
                                operand.role, operand.object, operand.tensor, operand.dtype,
                            ));
                        }
                    }
                    "ffn" => {
                        let f = &layer.ffn;
                        out.push(format!(
                            "    ffn        kind {}{}",
                            f.kind,
                            f.experts
                                .map_or(String::new(), |(e, k)| format!("  experts {e} top_k {k}")),
                        ));
                        for operand in &f.operands {
                            out.push(format!(
                                "      {:12} {}::{} @{}",
                                operand.role, operand.object, operand.tensor, operand.dtype,
                            ));
                        }
                    }
                    other => out.push(format!("    {other}")),
                }
            }
        }
        out.push(String::new());
        out.push("CONTINUATION".into());
        out.push(format!(
            "  provider: {} (LQL's declared continuation)",
            CanonicalFactory.identity()
        ));
        for (layer, g) in explain.continuation.iter().enumerate() {
            out.push(format!(
                "  layer {layer}: kv_dim {} window {}",
                g.kv_dim,
                g.window.map_or("-".into(), |w| w.to_string()),
            ));
        }
        out.push(String::new());
        out.push("OUTPUT".into());
        match &explain.output {
            Some(head) => {
                out.push(format!(
                    "  output_head: present  vocab {}{}{}",
                    head.vocab,
                    if head.multiplied { "  multiplied" } else { "" },
                    if head.softcapped { "  softcapped" } else { "" },
                ));
            }
            None => out.push("  output_head: absent".into()),
        }
        out.push(format!(
            "  final_norm: {}",
            if explain.final_norm {
                "present"
            } else {
                "absent"
            }
        ));
        Ok(out)
    }

    /// `TRACE "prompt"` on a V3 binding (LQL-2): observe the canonical
    /// executor while it ingests the prompt, then report the greedy
    /// next token. Observation is subscription — the executor's parity
    /// gate pins that tracing never changes arithmetic, and the LQL
    /// gate pins that the reported token equals INFER's.
    pub(crate) fn exec_v3_trace(&self, prompt: &str) -> Result<Vec<String>, LqlError> {
        use larql_inference::vindex3::{CarrierForm, RecordingObserver, StepEvent, SublayerSite};
        let Backend::Vindex3 {
            runtime,
            tokenizer,
            overlay,
            bos_token,
            ..
        } = &self.backend
        else {
            unreachable!("caller matched the backend");
        };
        let tokenizer = tokenizer.as_ref().ok_or_else(|| {
            LqlError::Execution("TRACE needs a tokenizer — this container carries none".into())
        })?;
        let prompt_ids = encode_raw_prompt(tokenizer, prompt, *bos_token)?;

        // TRACE observes the same effective program INFER runs — a
        // compose edit must not fork the two.
        let continuation = v3_continuation(runtime)?;
        let mut session = match compose_overrides(runtime, overlay)? {
            Some(overrides) => runtime.session_overlaid(&overrides, &continuation),
            None => runtime.session(&continuation),
        }
        .map_err(|e| LqlError::exec("v3 session failed", e))?;
        let mut out = vec!["Trace (VINDEX3 program, observed execution):".into()];
        let mut logits = Vec::new();
        for (offset, &token) in prompt_ids.iter().enumerate() {
            let mut recorder = RecordingObserver::default();
            logits = session
                .step_observed(token, &mut recorder)
                .map_err(|e| LqlError::exec("v3 step failed", e))?;
            out.push(format!("position {offset} (prompt token {token})"));
            for event in &recorder.events {
                match event {
                    StepEvent::Embedded { .. } => out.push("  embed".into()),
                    StepEvent::AttentionDone { layer } => {
                        out.push(format!("  layer {layer}: attention"))
                    }
                    StepEvent::FfnDone { layer } => out.push(format!("  layer {layer}: ffn")),
                    StepEvent::CarrierWrite {
                        layer,
                        site,
                        carrier,
                    } => {
                        let site = match site {
                            SublayerSite::Attention => "attention",
                            SublayerSite::Ffn => "ffn",
                        };
                        let carrier = match carrier {
                            CarrierForm::Single => "single",
                            CarrierForm::Bundle => "bundle",
                            CarrierForm::History => "history",
                        };
                        out.push(format!("  layer {layer}: {site} write ({carrier} carrier)"))
                    }
                    StepEvent::Logits { vocab } => {
                        out.push(format!("  output_head (vocab {vocab})"))
                    }
                    // The executor may learn new events before TRACE
                    // learns to print them; an unprinted event is not
                    // an error.
                    _ => {}
                }
            }
        }
        let mut sampler = larql_inference::Sampler::new(SamplingConfig::greedy());
        let next = sampler
            .sample(&logits)
            .ok_or_else(|| LqlError::Execution("no next token from the logits".into()))?;
        let text = tokenizer.decode(&[next], false).unwrap_or_default();
        out.push(format!("next token {next} {text:?} (greedy)"));
        Ok(out)
    }
}

impl Session {
    /// The bound container root for the directory statements (`SHOW
    /// COMPONENTS/REPRESENTATIONS/PROVENANCE/AUTHORITY`), or the
    /// refusal. These statements read the container's own declarations
    /// — system graph, representation directory, authority record —
    /// which only a VINDEX3 container carries.
    pub(super) fn v3_container_root(&self, what: &str) -> Result<&std::path::Path, LqlError> {
        match &self.backend {
            Backend::Vindex3 { path, .. } => Ok(path),
            Backend::None => Err(LqlError::NoBackend),
            _ => Err(LqlError::Execution(format!(
                "{what} reads a container's own declarations (system graph, \
                 representation directory, authority record) — VINDEX3 concepts. \
                 This binding is not a VINDEX3 container."
            ))),
        }
    }

    /// The container's index, read fresh from disk. The directory
    /// statements answer from the container's declarations, not from
    /// any runtime state — the same source `larql show` reads.
    pub(super) fn v3_index(
        &self,
        what: &str,
    ) -> Result<larql_vindex::format::vindex3::index::Vindex3Index, LqlError> {
        let root = self.v3_container_root(what)?;
        let text = std::fs::read_to_string(root.join(larql_vindex::format::filenames::INDEX_JSON))
            .map_err(|e| LqlError::exec("read container index", e))?;
        serde_json::from_str(&text).map_err(|e| LqlError::exec("parse container index", e))
    }

    /// `SHOW COMPONENTS`: the system graph's components, reconstructed
    /// purely from the container — id, role, and geometry as declared.
    pub(crate) fn exec_show_components(&self) -> Result<Vec<String>, LqlError> {
        let root = self.v3_container_root("SHOW COMPONENTS")?;
        let inspection = larql_vindex::format::vindex3::inspect::inspect_container(root, false)
            .map_err(|e| LqlError::exec("inspect container", e))?;

        let mut out = Vec::new();
        out.push(format!(
            "{:<20} {:<16} {:>8} {:>8}  {}",
            "Component", "Role", "Layers", "Hidden", "Attention"
        ));
        out.push("-".repeat(70));
        for c in &inspection.components {
            let mut attention = Vec::new();
            if let Some(n) = c.full_layers.filter(|n| *n > 0) {
                attention.push(format!("{n} full"));
            }
            if let Some(n) = c.sliding_layers.filter(|n| *n > 0) {
                match c.window {
                    Some(w) => attention.push(format!("{n} sliding (window {w})")),
                    None => attention.push(format!("{n} sliding")),
                }
            }
            if let Some(n) = c.recurrent_layers.filter(|n| *n > 0) {
                attention.push(format!("{n} recurrent"));
            }
            let attention = if attention.is_empty() {
                "-".to_string()
            } else {
                attention.join(" / ")
            };
            out.push(format!(
                "{:<20} {:<16} {:>8} {:>8}  {}",
                c.id, c.role, c.num_layers, c.hidden_size, attention
            ));
        }
        out.push(format!(
            "{} object(s), {} hidden-state edge(s) — every fact above is graph data, \
             not a name convention",
            inspection.graph.objects.len(),
            inspection.graph.edges.len(),
        ));
        if !inspection.is_coherent() {
            out.push(format!(
                "warning: {} coherence defect(s) — run `larql vindex3 inspect` for the report",
                inspection.defects.len(),
            ));
        }
        Ok(out)
    }

    /// `SHOW REPRESENTATIONS ["object"]`: the physically present
    /// representation directory, optionally filtered by object id
    /// substring. Presence is physical: a variant listed here exists as
    /// bytes; selecting one not listed fails closed before a byte is
    /// read (§9.1).
    pub(crate) fn exec_show_representations(
        &self,
        object: Option<&str>,
    ) -> Result<Vec<String>, LqlError> {
        let index = self.v3_index("SHOW REPRESENTATIONS")?;
        let mut out = Vec::new();
        out.push(format!(
            "{:<32} {:<24} {:<14} {:>8} {:>12}",
            "Representation", "Object", "Encoding", "Tensors", "Bytes"
        ));
        out.push("-".repeat(94));
        let mut shown = 0usize;
        for (id, entry) in &index.representations {
            if let Some(filter) = object {
                if !entry.object.contains(filter) && !id.contains(filter) {
                    continue;
                }
            }
            let compiled = if entry.compiled_from.is_some() {
                " (compiled)"
            } else {
                ""
            };
            out.push(format!(
                "{:<32} {:<24} {:<14} {:>8} {:>12}{}",
                id, entry.object, entry.encoding, entry.tensor_count, entry.payload_bytes, compiled,
            ));
            shown += 1;
        }
        if shown == 0 {
            out.push(match object {
                Some(filter) => format!("  (no representations match \"{filter}\")"),
                None => "  (the directory is empty)".into(),
            });
        }
        Ok(out)
    }

    /// `SHOW PROVENANCE ["object"]`: hashes and lineage per directory
    /// entry. The digests are printed whole — provenance abbreviated is
    /// provenance lost.
    pub(crate) fn exec_show_provenance(
        &self,
        object: Option<&str>,
    ) -> Result<Vec<String>, LqlError> {
        let index = self.v3_index("SHOW PROVENANCE")?;
        let mut out = Vec::new();
        if let Some(model) = index.derived_from_model.as_deref() {
            out.push(format!("Container derives from: {model}"));
            out.push(String::new());
        }
        let mut shown = 0usize;
        for (id, entry) in &index.representations {
            if let Some(filter) = object {
                if !entry.object.contains(filter) && !id.contains(filter) {
                    continue;
                }
            }
            out.push(id.clone());
            out.push(format!("  object:          {}", entry.object));
            out.push(format!("  segment:         {}", entry.segment));
            out.push(format!("  payload_sha256:  {}", entry.payload_sha256));
            out.push(format!("  segment_sha256:  {}", entry.segment_sha256));
            match entry.compiled_from.as_deref() {
                Some(source) => out.push(format!("  compiled_from:   {source}")),
                None => out.push(
                    "  compiled_from:   (source checkpoint — no earlier container-side authority)"
                        .into(),
                ),
            }
            if let Some(digest) = entry.source_representation_digest.as_deref() {
                out.push(format!("  source_digest:   {digest}"));
            }
            shown += 1;
        }
        if shown == 0 {
            out.push(match object {
                Some(filter) => format!("  (no entries match \"{filter}\")"),
                None => "  (the directory is empty)".into(),
            });
        }
        Ok(out)
    }

    /// `SHOW AUTHORITY`: the container's own authority declaration —
    /// canonical (bit-authority present, derived representations
    /// recompilable) or derived (executable; not re-compilable) — and
    /// the profiles it declares by name.
    pub(crate) fn exec_show_authority(&self) -> Result<Vec<String>, LqlError> {
        use larql_vindex::format::vindex3::index::ContainerAuthority;
        let index = self.v3_index("SHOW AUTHORITY")?;
        let mut out = Vec::new();
        match index.authority {
            ContainerAuthority::Canonical => {
                out.push("Authority:   canonical".into());
                out.push(
                    "             source bytes present; derived representations can be recompiled"
                        .into(),
                );
            }
            ContainerAuthority::Derived => {
                out.push("Authority:   derived  (executable; not re-compilable)".into());
                if let Some(model) = index.derived_from_model.as_deref() {
                    out.push(format!("Source:      {model}"));
                }
            }
        }
        let profiles: Vec<&str> = index.profiles.iter().map(|p| p.name.as_str()).collect();
        if !profiles.is_empty() {
            out.push(format!("Profiles:    {}", profiles.join(", ")));
            out.push(
                "             a profile selects among physically present variants; \
                 selecting an absent one fails closed"
                    .into(),
            );
        }
        Ok(out)
    }
}

/// Open a container as a V3 binding: runtime (refusing closure
/// defects) plus the optional tokenizer capability.
pub(crate) type V3Knowledge = larql_vindex::format::vindex3::knowledge::KnowledgeView;

pub(crate) fn bind(
    path: &std::path::Path,
    source: RepresentationSource,
) -> Result<(V3Runtime, Option<Tokenizer>, Option<V3Knowledge>), LqlError> {
    // The canonical program (`want: None`), under the session's source. Spelled out and not
    // `OpenPolicy::default()`, whose source permits the loader to manufacture a lossy residency.
    let runtime = Vindex3Runtime::open_with_via(
        path,
        V3_COMPONENT,
        std::sync::Arc::new(LoweringRegistry::shipped()),
        &LoweringIdentity::cpu_production(),
        OpenPolicy { want: None, source },
    )
    .map_err(|e| LqlError::exec("failed to open VINDEX3 container", e))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(path).ok();
    // The browse view needs the tokenizer (feature annotations decode
    // token ids); a tokenizer-less container binds without it.
    let knowledge = match &tokenizer {
        Some(tok) => Some(
            runtime
                .knowledge_view(tok)
                .map_err(|e| LqlError::exec("failed to bind the V3 query surface", e))?,
        ),
        None => None,
    };
    Ok((runtime, tokenizer, knowledge))
}
