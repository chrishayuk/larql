## ADDED Requirements

### Requirement: get_layer MUST auto-promote compressed layers

`KvCache::get_layer(&mut self, layer)` SHALL transparently
populate the FP32 `layers[layer]` slot from `quantized_kv[layer]`
when the FP32 slot is empty and the compressed side-table is
populated. The compressed side-table SHALL remain populated after
the promote so that subsequent compressions don't have to re-encode
the unchanged data.

#### Scenario: get_layer returns FP32 after compress
- **WHEN** a layer is compressed via `quantize_layer` and then read via `get_layer`
- **THEN** the call SHALL return `Some(SharedKV)` (not `None`)
<!-- test: unbacked -->

#### Scenario: get_layer caches the promoted copy
- **WHEN** the same compressed layer is read twice via `get_layer`
- **THEN** the second read SHALL not invoke `dequantize_v_with_inverse_rotation` again (verified by checking the metric counter)
<!-- test: unbacked -->

### Requirement: get_layer_lazy MUST be the explicit-no-promote variant

`KvCache::get_layer_lazy(&self, layer) -> Option<&SharedKV>` SHALL
return `None` for compressed layers regardless of compressed-side-
table state. It SHALL NOT mutate the cache. Snapshots and
diagnostics use this variant.

#### Scenario: get_layer_lazy never promotes
- **WHEN** a layer is compressed and read via `get_layer_lazy`
- **THEN** the call SHALL return `None` and the FP32 slot SHALL remain empty
<!-- test: unbacked -->

### Requirement: promote_on_read_count MUST increment per promote

`KvCache::promote_on_read_count: AtomicU64` SHALL increment on
each auto-promote. Reads via `get_layer` that hit an already-
promoted layer SHALL NOT increment the counter (it tracks
"actual dequants performed," not "compressed-layer hits").

#### Scenario: counter increments only on first promote
- **WHEN** `get_layer` is called twice on a compressed layer
- **THEN** the counter SHALL increment by exactly 1
<!-- test: unbacked -->

### Requirement: clear_layer MUST also clear compressed side-table

`clear_layer(layer)` SHALL set both `layers[layer]` and
`quantized_kv[layer]` to `None`. A subsequent `get_layer` SHALL
return `None`.

#### Scenario: clear erases both storages
- **WHEN** a compressed layer is cleared via `clear_layer`
- **THEN** `is_layer_compressed` SHALL return `false` and `get_layer` SHALL return `None`
<!-- test: unbacked -->
