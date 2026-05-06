## ADDED Requirements

### Requirement: `larql run` dispatches one-shot, chat, expert, and remote-FFN modes

The `larql run` subcommand SHALL accept a model identifier plus an
optional prompt and MUST dispatch into one of four modes based on the
flag set: one-shot generation when a prompt is supplied, interactive
chat when the prompt is omitted, virtual-experts dispatch when
`--experts` is set, and remote-FFN routing when `--ffn URL` is
supplied. Strategy selection (Metal vs CPU, Q4K vs F32) MUST be
derived from the resolved vindex's quantisation format and the
`--metal` flag rather than from user-facing strings.

#### Scenario: `--help` advertises the experts dispatch flags
- **WHEN** the user runs `larql run --help`
- **THEN** stdout SHALL list both `--experts` and `--experts-dir`
<!-- test: larql_cli::test_run_experts::run_help_lists_experts_flags -->

#### Scenario: Unresolvable model name fails before any inference setup
- **WHEN** `larql run` is invoked with a nonsense model identifier
- **THEN** the process SHALL exit non-zero with stderr that mentions the unresolved name
<!-- test: larql_cli::test_run_experts::experts_with_bogus_model_path_errors_cleanly -->

#### Scenario: `--experts-dir` validates existence before loading the model
- **WHEN** `larql run --experts --experts-dir /nonexistent` is invoked against a real cached vindex
- **THEN** the process SHALL exit non-zero with stderr containing `--experts-dir does not exist`
<!-- test: larql_cli::test_run_experts::experts_dir_override_validates_existence -->

#### Scenario: Q4K vindex with Metal picks the Metal Q4K strategy
- **WHEN** `pick_strategy` is called with a Q4K vindex and Metal available
- **THEN** the chosen strategy SHALL be Metal Q4K
<!-- test: larql_cli::commands::primary::run_cmd::experts::tests::pick_strategy_q4k_with_metal_picks_metal -->

#### Scenario: Q4K vindex without Metal falls back to CPU Q4K
- **WHEN** `pick_strategy` is called with a Q4K vindex and Metal unavailable
- **THEN** the chosen strategy SHALL be CPU Q4K
<!-- test: larql_cli::commands::primary::run_cmd::experts::tests::pick_strategy_q4k_without_metal_picks_cpu_q4k -->

#### Scenario: Non-Q4K vindex falls back to CPU F32 even when Metal is requested
- **WHEN** `pick_strategy` is called with an unquantised vindex and Metal available
- **THEN** the chosen strategy SHALL be CPU F32 because Metal has no f32 path
<!-- test: larql_cli::commands::primary::run_cmd::experts::tests::pick_strategy_non_q4k_with_metal_falls_back_to_f32 -->
<!-- test: larql_cli::commands::primary::run_cmd::experts::tests::pick_strategy_non_q4k_without_metal_picks_cpu_f32 -->

#### Scenario: End-to-end experts chat dispatch round-trips a prompt
- **WHEN** `larql run <vindex> --experts --experts-dir <wasm>` is fed a prompt over stdin (ignored test, requires real model + GPU)
- **THEN** the REPL SHALL print dispatch evidence (`"op":` JSON or `op-call` notice) on stdout or stderr before exiting cleanly
<!-- test: larql_cli::test_run_experts::experts_chat_mode_dispatches_via_stdin -->

### Requirement: Cache resolution covers HF cache, local registry, and shorthand

The cache resolver MUST scan both the HuggingFace hub layout
(`datasets--owner--name/snapshots/...`) and the local registry
(`<root>/<name>.vindex/`), MUST list local entries before HF entries
when both are present, and SHALL accept four input forms:
`hf://owner/name`, `owner/name`, the bare name as a unique HF
shorthand, and a bare name as a local registration. Ambiguous
shorthands SHALL fail with an error that lists every candidate and
its source tag.

#### Scenario: Empty / missing cache directory returns an empty list
- **WHEN** `scan_hf_hub_at` and `scan_local_at` are called on a non-existent path
- **THEN** both SHALL return an empty `Vec` rather than erroring
<!-- test: larql_cli::commands::primary::cache::tests::scan_returns_empty_for_missing_dir -->
<!-- test: larql_cli::commands::primary::cache::tests::scan_local_empty_when_dir_missing -->

#### Scenario: HF scan finds and alphabetically sorts cached vindexes
- **WHEN** a fake HF cache contains three repos
- **THEN** `scan_hf_hub_at` SHALL return them sorted by repo name and tagged `CacheSource::HuggingFace`
<!-- test: larql_cli::commands::primary::cache::tests::scan_finds_cached_vindexes_and_sorts -->

#### Scenario: Snapshots without `index.json` are skipped
- **WHEN** the HF scan encounters a snapshot folder containing no `index.json`
- **THEN** the entry SHALL be omitted from the returned list
<!-- test: larql_cli::commands::primary::cache::tests::scan_skips_snapshots_without_index_json -->

#### Scenario: Local scan resolves symlinked vindex directories
- **WHEN** a local cache contains a symlink to a vindex directory
- **THEN** `scan_local_at` SHALL follow the symlink and report the target as a `CacheSource::Local` entry
<!-- test: larql_cli::commands::primary::cache::tests::scan_local_resolves_symlinks -->

#### Scenario: Merged scan emits HF entries before local entries
- **WHEN** both cache roots contain matching repos
- **THEN** `scan_cached_vindexes_at_both` SHALL return HF entries before local entries (the documented merge order)
<!-- test: larql_cli::commands::primary::cache::tests::scan_both_merges_and_orders_local_first -->

#### Scenario: Shorthand resolution surfaces both cache sources on ambiguity
- **WHEN** the same bare name resolves in both HF and local caches
- **THEN** `resolve_shorthand_from` SHALL fail with an error containing `ambiguous`, `[hf]`, and `[local]`
<!-- test: larql_cli::commands::primary::cache::tests::shorthand_ambiguous_across_hf_and_local_errors_with_sources -->

#### Scenario: Shorthand miss directs the user at `pull` and `link`
- **WHEN** `resolve_shorthand_from` is called with a name that matches no cache entry
- **THEN** the error message SHALL mention both `larql pull` and `larql link`
<!-- test: larql_cli::commands::primary::cache::tests::shorthand_no_match_mentions_both_registration_paths -->

#### Scenario: `owner/name`, `hf://`, and bare shorthand all resolve cached repos
- **WHEN** `resolve_cached_from` is called with `owner/name`, `hf://owner/name`, the bare HF shorthand, or a bare local shorthand
- **THEN** the resolver SHALL return the matching `CachedVindex` with the correct source tag
<!-- test: larql_cli::commands::primary::cache::tests::resolve_cached_accepts_owner_slash_name -->
<!-- test: larql_cli::commands::primary::cache::tests::resolve_cached_strips_hf_scheme -->
<!-- test: larql_cli::commands::primary::cache::tests::resolve_cached_accepts_hf_shorthand -->
<!-- test: larql_cli::commands::primary::cache::tests::resolve_cached_accepts_local_shorthand -->

### Requirement: `larql pull` accepts only HF identifiers and renders sibling slice repos

`larql pull` SHALL accept `hf://owner/name` and `owner/name`,
SHALL reject local paths and single-word names, and MUST render
sibling slice repository names from a configurable template so that
sibling presets stay consistent with the slice presets accepted by
`larql publish`. The default template SHALL be `{repo}-{preset}`.

#### Scenario: Default template renders `<repo>-<preset>`
- **WHEN** `render_sibling_repo` is called with a real HF repo and the `client` preset under the default template
- **THEN** the rendered repo SHALL equal `<owner>/<name>-client`
<!-- test: larql_cli::commands::primary::pull_cmd::tests::render_sibling_uses_default_template -->

#### Scenario: `hf://` prefix is stripped when rendering sibling repos
- **WHEN** the source repo carries an `hf://` scheme prefix
- **THEN** the rendered sibling SHALL drop the scheme so the result is a plain `owner/name`
<!-- test: larql_cli::commands::primary::pull_cmd::tests::render_sibling_strips_hf_prefix -->

#### Scenario: Local paths are rejected as sibling sources
- **WHEN** `render_sibling_repo` is called with a local filesystem path
- **THEN** it SHALL fail with an error mentioning `owner/name`
<!-- test: larql_cli::commands::primary::pull_cmd::tests::render_sibling_rejects_local_path -->

#### Scenario: HF normaliser accepts both `hf://` and bare `owner/name`
- **WHEN** `normalise_hf_path` is called with either form
- **THEN** the returned value SHALL canonicalise to `hf://owner/name`
<!-- test: larql_cli::commands::primary::pull_cmd::tests::normalise_hf_path_accepts_hf_prefix_and_owner_name -->

#### Scenario: HF normaliser rejects single-word and absolute paths
- **WHEN** `normalise_hf_path` is called with a bare name or absolute path
- **THEN** it SHALL return an error
<!-- test: larql_cli::commands::primary::pull_cmd::tests::normalise_hf_path_rejects_single_word -->
<!-- test: larql_cli::commands::primary::pull_cmd::tests::normalise_hf_path_rejects_local_path -->

#### Scenario: Sibling-suffix split mirrors publish preset names
- **WHEN** `split_sibling_suffix` is called on a known sibling repo
- **THEN** it SHALL return the base repo and the recognised preset suffix
<!-- test: larql_cli::commands::primary::pull_cmd::tests::split_sibling_suffix_recognises_known_presets -->
<!-- test: larql_cli::commands::primary::pull_cmd::tests::split_sibling_suffix_leaves_full_repo_untouched -->

#### Scenario: Pull and publish keep sibling preset lists in sync
- **WHEN** the sibling-preset list is compared against `publish_cmd::DEFAULT_SLICES`
- **THEN** the two lists SHALL be equal so a `pull` hint always matches what `publish` produced
<!-- test: larql_cli::commands::primary::pull_cmd::tests::default_sibling_presets_match_publish_defaults -->

### Requirement: `larql slice` carves the right parts per deployment preset

`larql slice` SHALL parse part names case-insensitively (with aliases
such as `attn`/`attention`, `embed`/`embed-server`), MUST recognise
quantised filenames as the same logical part as their unquantised
counterparts (e.g. `attn_weights_q4k.bin` matches the `Attn` part),
and MUST resolve every preset (`client`, `server`, `attn`, `embed`,
`router`, `browse`) into a part set whose effective `ExtractLevel`
matches the deployment shape the preset describes.

#### Scenario: Part-name aliases resolve to the same part
- **WHEN** `Part::parse` is called on `"attn"`, `"attention"`, or `"Embeddings"`
- **THEN** each SHALL return the same canonical `Part` value, and unknown names SHALL return `None`
<!-- test: larql_cli::commands::primary::slice_cmd::tests::part_parse_aliases -->

#### Scenario: Attention part matches both float and Q4K filenames
- **WHEN** `Part::Attn::matches` is queried against `attn_weights.bin`, `attn_weights_q4_*.bin`, and `attn_weights_q4k_manifest.json`
- **THEN** every variant SHALL match, and `gate_vectors.bin` SHALL NOT
<!-- test: larql_cli::commands::primary::slice_cmd::tests::attn_matches_quant_variants -->

#### Scenario: FFN part matches interleaved + hidden-major filenames
- **WHEN** `Part::Ffn::matches` is queried against `interleaved.bin`, `interleaved_q4k.bin`, `up_weights.bin`, and `down_features.bin`
- **THEN** every variant SHALL match, while gate vectors stay outside the FFN part
<!-- test: larql_cli::commands::primary::slice_cmd::tests::ffn_matches_interleaved_and_hidden_major -->

#### Scenario: `client` preset is the 2-tier attention slice without FFN bytes
- **WHEN** `preset_parts("client")` is read
- **THEN** the result SHALL contain attention, norms, embed, and tokenizer, and MUST NOT contain FFN compute weights
<!-- test: larql_cli::commands::primary::slice_cmd::tests::preset_client_is_attention_tier -->

#### Scenario: `server` preset carries FFN + gate but not attention
- **WHEN** `preset_parts("server")` is read
- **THEN** it SHALL contain FFN, gate, down-meta, embed, and norms but MUST omit attention
<!-- test: larql_cli::commands::primary::slice_cmd::tests::preset_server_carries_ffn_not_attention -->

#### Scenario: `attn` preset is attention-only (no embed, no tokenizer)
- **WHEN** `preset_parts("attn")` is read
- **THEN** the result SHALL contain attention + norms but MUST NOT contain embed, gate, FFN, or tokenizer
<!-- test: larql_cli::commands::primary::slice_cmd::tests::preset_attn_is_attention_without_embed -->

#### Scenario: `embed` preset is embed-server scope (no attention, no norms)
- **WHEN** `preset_parts("embed")` is read
- **THEN** the result SHALL contain embed + tokenizer only
<!-- test: larql_cli::commands::primary::slice_cmd::tests::preset_embed_carries_embed_and_tokenizer_only -->

#### Scenario: Unknown preset surfaces an error
- **WHEN** `preset_parts("xyz")` is called
- **THEN** it SHALL return an error rather than an empty / default set
<!-- test: larql_cli::commands::primary::slice_cmd::tests::preset_unknown_errors -->

#### Scenario: Effective extract level reflects the chosen parts
- **WHEN** the part set is reduced to attention + norms + embed + tokenizer
- **THEN** `effective_level` SHALL report `ExtractLevel::Attention`
<!-- test: larql_cli::commands::primary::slice_cmd::tests::effective_level_client_is_attention -->

#### Scenario: Server slice without attention caps at Browse
- **WHEN** the part set lacks attention but contains FFN bytes
- **THEN** `effective_level` SHALL cap at `ExtractLevel::Browse`
<!-- test: larql_cli::commands::primary::slice_cmd::tests::effective_level_server_is_browse_without_attn -->

#### Scenario: Effective level cannot exceed the source level
- **WHEN** a full part set is asked to elevate above a Browse-only source
- **THEN** `effective_level` SHALL stay at `ExtractLevel::Browse`
<!-- test: larql_cli::commands::primary::slice_cmd::tests::effective_level_capped_by_source -->

### Requirement: `larql publish` builds the documented HF layout

`larql publish` SHALL upload the source vindex plus a configurable
list of slice siblings, MUST publish `model`, `family`, and `library`
collection levels by default, MUST honour `--force-upload` to disable
the SHA256 skip-if-unchanged check, and SHALL render every published
slice's repo name from a `{repo}` / `{preset}` template. Bare-name
sources (without an `owner/`) SHALL be rejected so we never silently
publish to the wrong namespace.

#### Scenario: Default slice list publishes the documented five presets
- **WHEN** `resolve_slice_list(&[])` is called
- **THEN** the result SHALL equal `["client", "attn", "embed", "server", "browse"]` in that order
<!-- test: larql_cli::commands::primary::publish_cmd::tests::default_slice_list_is_full_publish_set -->

#### Scenario: `none` (case-insensitive) disables sliced uploads
- **WHEN** `resolve_slice_list(&["none"])` or `&["NONE"]` is called
- **THEN** the returned list SHALL be empty
<!-- test: larql_cli::commands::primary::publish_cmd::tests::slices_none_disables_sliced_uploads -->

#### Scenario: Explicit slice presets pass through unchanged
- **WHEN** `resolve_slice_list(&["client", "server"])` is called
- **THEN** the result SHALL equal the input list
<!-- test: larql_cli::commands::primary::publish_cmd::tests::slices_explicit_list_passes_through -->

#### Scenario: `router` preset is accepted even though it is not default
- **WHEN** the user explicitly requests the `router` slice
- **THEN** `resolve_slice_list` SHALL accept it
<!-- test: larql_cli::commands::primary::publish_cmd::tests::slices_with_router_is_valid -->

#### Scenario: Unknown slice name fails with a helpful error
- **WHEN** `resolve_slice_list(&["typo"])` is called
- **THEN** it SHALL fail with an error mentioning `invalid slice preset`
<!-- test: larql_cli::commands::primary::publish_cmd::tests::slices_invalid_name_errors -->

#### Scenario: Default collection levels publish to model/family/library
- **WHEN** the publish CLI defaults are passed through `resolve_collection_list`
- **THEN** the returned list SHALL equal `["model", "family", "library"]`
<!-- test: larql_cli::commands::primary::publish_cmd::tests::default_collection_levels_are_all_three -->

#### Scenario: Collection level `none` disables the document tree
- **WHEN** `resolve_collection_list(&["none"])` (any case) is called
- **THEN** the result SHALL be empty
<!-- test: larql_cli::commands::primary::publish_cmd::tests::collection_level_none_disables_all -->

#### Scenario: Collection levels are lowercased before validation
- **WHEN** mixed-case level names are supplied
- **THEN** they SHALL be normalised to lower-case before being matched
<!-- test: larql_cli::commands::primary::publish_cmd::tests::collection_level_is_lowercased -->

#### Scenario: Bare-name source rejected; `owner/name` accepted
- **WHEN** `namespace_of` is called on `gemma-4-31b` vs `chrishayuk/gemma-4-31b`
- **THEN** the bare name SHALL fail and the qualified name SHALL return `chrishayuk`
<!-- test: larql_cli::commands::primary::publish_cmd::tests::namespace_of_rejects_bare_name -->

#### Scenario: Default model title strips HF cache layout
- **WHEN** `default_model_title` is called on either an `owner/name` repo or an HF cache absolute path
- **THEN** the rendered title SHALL be human-readable (e.g. `Gemma 4 31b It`) and never empty
<!-- test: larql_cli::commands::primary::publish_cmd::tests::default_model_title_strips_hf_namespace -->
<!-- test: larql_cli::commands::primary::publish_cmd::tests::default_model_title_from_hf_cache_path -->
<!-- test: larql_cli::commands::primary::publish_cmd::tests::short_model_name_handles_hf_cache_layout -->

#### Scenario: Default family stops at the first version-style segment
- **WHEN** `default_family` is called on `gemma-3-4b-it` / `llama-3-8b-instruct`
- **THEN** the family SHALL be the title-cased prefix up to the first digit-leading segment
<!-- test: larql_cli::commands::primary::publish_cmd::tests::default_family_stops_at_first_digit_segment -->
<!-- test: larql_cli::commands::primary::publish_cmd::tests::default_family_multi_word_prefix_preserved -->
<!-- test: larql_cli::commands::primary::publish_cmd::tests::default_family_no_digit_title_cases_all_segments -->

#### Scenario: Every default slice has a hand-written collection note
- **WHEN** `note_for_preset` is called on every default slice plus `router`
- **THEN** each note SHALL describe the deployment shape the preset implies
<!-- test: larql_cli::commands::primary::publish_cmd::tests::note_for_preset_covers_every_default_slice -->

#### Scenario: `--force-upload` disables the unchanged-hash skip
- **WHEN** `PublishOptions { skip_unchanged: false, .. }` is constructed (the `--force-upload` shape)
- **THEN** `skip_unchanged` SHALL be `false`
<!-- test: larql_cli::commands::primary::publish_cmd::tests::force_upload_disables_skip -->

#### Scenario: Default publish skips already-uploaded files
- **WHEN** `PublishOptions { skip_unchanged: true, .. }` is constructed (the default)
- **THEN** `skip_unchanged` SHALL be `true`
<!-- test: larql_cli::commands::primary::publish_cmd::tests::default_publish_options_skip_unchanged -->

### Requirement: Diagnostic and utility commands gate stale formats

The diagnostic commands SHALL validate Q4K stride manifests against
the canonical 144-byte super-block size and MUST treat the legacy
148-byte stride as a fatal error so users rebuild stale vindexes.
`larql shannon` MUST round-trip both arithmetic-coded `ShannonFile`
payloads and per-block vindex shannon payloads. Byte counts SHALL
render with the documented unit ladder (B / KB / MB / GB).

#### Scenario: Canonical 144-byte Q4_K stride passes diagnostic validation
- **WHEN** `validate_strides` is run against a manifest with `length = rows * blocks * 144`
- **THEN** it SHALL return a string starting with the `✓` checkmark
<!-- test: larql_cli::commands::primary::diag_cmd::tests::validate_strides_accepts_canonical_144_byte -->

#### Scenario: Legacy 148-byte stride is reported as STALE
- **WHEN** `validate_strides` is run against a manifest using the legacy 148-byte stride
- **THEN** it SHALL fail with an error mentioning both `stale` and `rebuild`
<!-- test: larql_cli::commands::primary::diag_cmd::tests::validate_strides_rejects_legacy_148_byte -->

#### Scenario: Mixed Q4_K + Q6_K manifest validates each format independently
- **WHEN** the manifest mixes Q4_K and Q6_K entries with their respective expected byte counts
- **THEN** validation SHALL succeed for both
<!-- test: larql_cli::commands::primary::diag_cmd::tests::validate_strides_handles_mixed_q4k_q6k -->

#### Scenario: Missing manifest is treated as zero entries, not an error
- **WHEN** `validate_strides` runs on a directory with no Q4K manifest
- **THEN** it SHALL return success
<!-- test: larql_cli::commands::primary::diag_cmd::tests::validate_strides_handles_missing_manifest -->

#### Scenario: Byte counts render with the documented unit ladder
- **WHEN** `human_size` is called with bytes spanning `B`, `KB`, `MB`, and `GB`
- **THEN** the rendered strings SHALL match the documented format
<!-- test: larql_cli::commands::primary::diag_cmd::tests::human_size_units -->

#### Scenario: Arithmetic encoder/decoder round-trips a fixed symbol stream
- **WHEN** a fixed-counts model is used to encode then decode a symbol stream via `ArithmeticEncoder`/`ArithmeticDecoder`
- **THEN** the decoded stream SHALL equal the original
<!-- test: larql_cli::commands::primary::shannon_cmd::tests::arithmetic_round_trip_fixed_counts -->

#### Scenario: `ShannonFile` survives the byte-level round-trip
- **WHEN** a `ShannonFile` is serialised to bytes and parsed back
- **THEN** all fields SHALL match the original
<!-- test: larql_cli::commands::primary::shannon_cmd::tests::shannon_file_round_trip -->

#### Scenario: Vindex shannon blocks survive the byte-level round-trip
- **WHEN** a vector of `VindexShannonBlock`s is encoded and decoded
- **THEN** every block field SHALL match the original, and a too-short input SHALL parse to `None`
<!-- test: larql_cli::commands::primary::shannon_cmd::tests::vindex_blocks_round_trip -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_cli::test_run_experts::**::* -->
<!-- test: larql_cli::commands::primary::cache::tests::**::* -->
<!-- test: larql_cli::commands::primary::pull_cmd::tests::**::* -->
<!-- test: larql_cli::commands::primary::publish_cmd::tests::**::* -->
<!-- test: larql_cli::commands::primary::slice_cmd::tests::**::* -->
<!-- test: larql_cli::commands::primary::diag_cmd::tests::**::* -->
<!-- test: larql_cli::commands::primary::shannon_cmd::tests::**::* -->
<!-- test: larql_cli::commands::primary::run_cmd::experts::tests::**::* -->
<!-- test: larql_cli::trampoline_tests::**::* -->
