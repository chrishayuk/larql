//! `EXPLAIN WALK` — verbose walk trace for a prompt.

use crate::ast::Range;
use crate::error::LqlError;
use crate::executor::Session;

impl Session {
    pub(crate) fn exec_explain(
        &self,
        prompt: &str,
        layers: Option<&Range>,
        verbose: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (path, config, patched) = self.require_vindex()?;

        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        let token_ids = super::encode_vindex_prompt(config, &tokenizer, prompt)?;

        if token_ids.is_empty() {
            return Err(LqlError::Execution("empty prompt".into()));
        }

        let last_tok = *token_ids.last().unwrap();
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

        let top_k = if verbose { 10 } else { 5 };
        let trace = patched.walk(&query, &walk_layers, top_k);

        let mut out = Vec::new();
        for (layer, hits) in &trace.layers {
            let show_count = if verbose {
                hits.len()
            } else {
                hits.len().min(5)
            };
            for hit in hits.iter().take(show_count) {
                let down_count = if verbose { 5 } else { 3 };
                let down_tokens: String = hit
                    .meta
                    .top_k
                    .iter()
                    .take(down_count)
                    .map(|t| t.token.clone())
                    .collect::<Vec<_>>()
                    .join(", ");

                out.push(format!(
                    "L{}: F{} → {} (gate={:.1}, down=[{}])",
                    layer, hit.feature, hit.meta.top_token, hit.gate_score, down_tokens
                ));
            }
        }

        Ok(out)
    }
}
