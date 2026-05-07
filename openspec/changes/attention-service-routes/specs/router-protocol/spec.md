## MODIFIED Requirements

### Requirement: AnnounceMsg SHALL carry a capabilities list

`AnnounceMsg` SHALL include `repeated string capabilities` (proto
field). The router SHALL accept the field; pre-extension shards
that omit the field SHALL be treated as if they sent
`["attention", "expert"]`.

#### Scenario: announce with explicit capabilities preserves the list

- **WHEN** a shard announces with `capabilities = ["attention"]`
- **THEN** `GridState::register` SHALL store exactly that vec on
  `ServerEntry::capabilities`
<!-- test: unbacked -->

#### Scenario: announce without capabilities defaults to both

- **WHEN** a shard announces with no `capabilities` field
- **THEN** `ServerEntry::capabilities` SHALL be
  `["attention", "expert"]`
<!-- test: unbacked -->

### Requirement: HeartbeatMsg SHALL carry the cached-prefix bloom

`HeartbeatMsg` SHALL include `bytes cached_prefixes` (proto field).
The field SHALL be exactly 32 bytes long when set, and SHALL be
absent (or empty) for shards that haven't been upgraded. Empty /
absent values SHALL be treated as the empty bloom (matches no
prefixes).

#### Scenario: heartbeat with bloom updates the entry

- **WHEN** a shard heartbeat carries a non-empty 32-byte
  `cached_prefixes`
- **THEN** `GridState::update_heartbeat` SHALL deserialise it into
  `PrefixBloom` and write it onto `ServerEntry::cached_prefixes`
<!-- test: unbacked -->

#### Scenario: heartbeat without bloom leaves prior value

- **WHEN** a shard heartbeat is received with no `cached_prefixes`
  field
- **THEN** the existing `ServerEntry::cached_prefixes` SHALL NOT be
  cleared (heartbeats partially update; pre-extension shards never
  populate the bloom but the field remains the empty default)
<!-- test: unbacked -->

#### Scenario: malformed bloom is rejected without panic

- **WHEN** a heartbeat carries a `cached_prefixes` of length other
  than 32
- **THEN** `update_heartbeat` SHALL ignore the field, log a
  warning, and update CPU/RAM/in-flight as usual
<!-- test: unbacked -->
