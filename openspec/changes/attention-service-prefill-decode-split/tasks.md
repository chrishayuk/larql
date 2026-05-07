## 1. --mode flag in larql-server

- [ ] 1.1 Add `--mode` to the CLI surface; values `prefill |
      decode | both` (default `both`).
- [ ] 1.2 Reject mismatched RPCs with HTTP 503 + structured body.

## 2. Stateless prefill route

- [ ] 2.1 `POST /v1/attention/prefill` runs a one-shot prefill,
      returns residuals + snapshot blob.
- [ ] 2.2 Verify server-side state is freed before response is sent.

## 3. Session restore from snapshot

- [ ] 3.1 `POST /v1/attention/session` accepts
      `restore_from_snapshot: <base64>`.
- [ ] 3.2 `current_length` post-restore equals the snapshot's row count.

## 4. Router sub-capability tags

- [ ] 4.1 `route_for_capability(_, _, "attention-prefill")` prefers
      that exact tag, falls back to `"attention"`.
- [ ] 4.2 Symmetric for `"attention-decode"`.

## 5. Tests

- [ ] 5.1 `mode_prefill_rejects_decode`.
- [ ] 5.2 `mode_decode_rejects_prefill`.
- [ ] 5.3 `mode_both_accepts_both`.
- [ ] 5.4 `prefill_response_includes_snapshot`.
- [ ] 5.5 `prefill_is_stateless`.
- [ ] 5.6 `session_restores_from_snapshot`.
- [ ] 5.7 `router_routes_to_attention_prefill_shard`.
- [ ] 5.8 `router_falls_back_to_catchall_attention`.

## 6. Validation

- [ ] 6.1 `openspec validate attention-service-prefill-decode-split --strict` passes.
- [ ] 6.2 Tests pass once `attention-service-routes` lands first.
- [ ] 6.3 `make traceability-check` and `make openspec-validate` pass.
- [ ] 6.4 `deploy/docker/README.md` documents the topology.
