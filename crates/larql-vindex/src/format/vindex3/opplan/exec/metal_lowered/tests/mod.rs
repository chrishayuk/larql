//! Gates for the lowered session: device-free helpers, and device tests
//! that run a whole `LoweredSession` on a fixture (they fail, never skip,
//! without a Metal device).

mod dense_layer_scale;
mod fixture;
mod gate_source;
mod profile;
mod rope_table;
