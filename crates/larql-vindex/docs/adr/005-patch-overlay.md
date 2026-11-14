# ADR-005: Patch Overlay for Editable Knowledge

**Status**: Accepted  
**Date**: 2026-04  
**Context**: Need to edit model knowledge without modifying the base vindex files.

## Decision

`PatchedVindex` wraps a readonly `VectorIndex` with an in-memory overlay. Mutations (insert, delete, update) go to the overlay. Base files are never modified.

```rust
let base = VectorIndex::load_vindex(&path, &mut callbacks)?;
let mut patched = PatchedVindex::new(base);

// Mutations go to overlay
patched.insert_feature(layer, feature, gate_vec, meta);
patched.delete_feature(layer, feature);

// Overlay serializable as a patch file
let patch = VindexPatch::from_operations(ops);
patch.save("medical.vlp")?;

// Patches stackable and reversible
patched.try_apply_patch(patch)?;
patched.remove_patch(0);

// Bake overlay into new clean base
let baked = patched.bake_down();
```

## Origin

Original LARQL design. Inspired by overlay filesystems (UnionFS) and database MVCC — base data is immutable, mutations are applied as a layer on top.

## Consequences

- Base vindex files can be shared/cached across users
- Patches are small (only changed features, not full model)
- Multiple patches composable (medical + legal + company knowledge)
- Reversible: remove a patch to undo its changes
- `bake_down()` creates a new clean base incorporating all patches
