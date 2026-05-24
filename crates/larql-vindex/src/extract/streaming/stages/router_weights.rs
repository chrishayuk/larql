//! Stage 1b — router weights (MoE models only).

use std::io::{BufWriter, Write};

use crate::error::VindexError;
use crate::extract::stage_labels::*;
use crate::extract::streaming::context::StreamingContext;
use crate::extract::streaming::tensor_io::normalize_key;
use crate::format::filenames::*;

impl<'a> StreamingContext<'a> {
    /// Stage 1b — router weights (MoE models only).
    pub(in crate::extract::streaming) fn write_router_weights(
        &mut self,
    ) -> Result<(), VindexError> {
        if !self.is_moe {
            return Ok(());
        }
        self.callbacks.on_stage(STAGE_ROUTER_WEIGHTS);
        let router_path = self.output_dir.join(ROUTER_WEIGHTS_BIN);
        let mut router_file = BufWriter::new(std::fs::File::create(&router_path)?);
        let prefixes: Vec<&str> = self.prefixes.iter().map(|s| s.as_str()).collect();

        for layer in 0..self.num_layers {
            let router_key = self
                .arch
                .moe_router_key(layer)
                .map(|k| normalize_key(&k, &prefixes))
                .unwrap_or_default();

            if let Some(tensor) = self.tensor_source.get_tensor_f32(&router_key)? {
                let data = tensor.as_slice().unwrap();
                let bytes = crate::config::dtype::encode_floats(data, self.dtype);
                router_file.write_all(&bytes)?;
            }

            // Also try router bias
            let bias_key = router_key.replace(".weight", ".bias");
            if let Some(tensor) = self.tensor_source.get_tensor_f32(&bias_key)? {
                let data = tensor.as_slice().unwrap();
                let bytes = crate::config::dtype::encode_floats(data, self.dtype);
                // Write bias after weight for each layer
                router_file.write_all(&bytes)?;
            }
        }
        router_file.flush()?;
        self.callbacks.on_stage_done(STAGE_ROUTER_WEIGHTS, 0.0);
        Ok(())
    }
}
