//! `WALK` — pure vindex feature scan, no attention.

use crate::ast::{Range, WalkMode};
use crate::error::LqlError;
use crate::executor::Session;

impl Session {
    pub(crate) fn exec_walk(
        &self,
        prompt: &str,
        top: Option<u32>,
        layers: Option<&Range>,
        mode: Option<WalkMode>,
        compare: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (path, config, patched) = self.require_vindex()?;
        let top_k = top.unwrap_or(10) as usize;

        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        let token_ids = super::encode_vindex_prompt(config, &tokenizer, prompt)?;

        if token_ids.is_empty() {
            return Err(LqlError::Execution("empty prompt".into()));
        }

        let last_tok = *token_ids.last().unwrap();
        let token_str = tokenizer
            .decode(&[last_tok], true)
            .unwrap_or_else(|_| format!("T{last_tok}"));

        let embed_row = embed.row(last_tok as usize);
        let query: larql_vindex::ndarray::Array1<f32> = embed_row.mapv(|v| v * embed_scale);

        let all_layers = patched.loaded_layers();
        let walk_layers: Vec<usize> = if let Some(range) = layers {
            (range.start as usize..=range.end as usize)
                .filter(|l| all_layers.contains(l))
                .collect()
        } else {
            all_layers
        };

        let start = std::time::Instant::now();
        let trace = patched.walk(&query, &walk_layers, top_k);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mode_str = match mode {
            Some(WalkMode::Pure) => "pure (sparse KNN only)",
            Some(WalkMode::Dense) => "dense (full matmul)",
            Some(WalkMode::Hybrid) | None => "hybrid (default)",
        };

        let mut out = Vec::new();
        out.push(format!(
            "Feature scan for {:?} (token {:?}, {} layers, mode={})",
            prompt,
            token_str.trim(),
            walk_layers.len(),
            mode_str,
        ));
        out.push(String::new());

        let show_per_layer = if compare { 5 } else { 3 };
        for (layer, hits) in &trace.layers {
            if hits.is_empty() {
                continue;
            }
            for hit in hits.iter().take(show_per_layer) {
                let down_top: String = hit
                    .meta
                    .top_k
                    .iter()
                    .take(3)
                    .map(|t| t.token.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push(format!(
                    "  L{:2}: F{:<5} gate={:+.1}  top={:15}  down=[{}]",
                    layer,
                    hit.feature,
                    hit.gate_score,
                    format!("{:?}", hit.meta.top_token),
                    down_top,
                ));
            }
        }

        out.push(format!("\n{:.1}ms", elapsed_ms));
        if compare {
            out.push(String::new());
            out.push(
                "Note: COMPARE shows more features per layer. For inference use INFER.".into(),
            );
        } else {
            out.push(String::new());
            out.push("Note: pure vindex scan (no attention). For inference use INFER.".into());
        }

        Ok(out)
    }
}
