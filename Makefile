.PHONY: build release test test-fast test-full test-integration test-models check clean fmt lint demos bench bench-save bench-check coverage coverage-summary coverage-check coverage-install ci-coverage traceability traceability-check gaps gaps-untested gaps-unbacked openspec-validate ci-cuda test-cuda docker-ffn docker-gpu docker-up docker-up-cpu docker-down docker-logs cuda-status attention-smoke attention-validate attention-validate-gemma attention-bench

# Build
build:
	cargo build --workspace

release:
	cargo build --release -p larql-cli

# Test
#
# Default test target is intentionally fast: no integration binaries, no
# model-backed ignored tests. Use `test-full` for the historical full
# workspace run, and `test-models` for real-model/vindex checks.
test: test-fast

test-fast:
	cargo test --workspace --lib --bins

test-full:
	cargo test --workspace

test-integration:
	cargo test --workspace --tests

test-models:
	cargo test -p larql-inference --test test_arch_golden -- --ignored
	cargo test -p larql-inference --test test_logits_goldens -- --ignored
	cargo test -p larql-inference --test test_gemma3_smoke -- --ignored
	cargo test -p larql-inference --test test_generate_q4k_cpu -- --ignored
	cargo test -p larql-inference --test bench_probe_latency -- --ignored --nocapture
	cargo test -p larql-inference --test test_llm_dispatch -- --ignored --nocapture
	cargo test -p larql-inference --test test_constrained_dispatch -- --ignored --nocapture
	cargo test -p larql-inference --test test_trie_dispatch -- --ignored --nocapture

# CUDA test suite — requires LARQL_CUDA_AVAILABLE=1 and a working CUDA
# runtime (driver + libcublas matching the cudarc feature in
# crates/larql-compute/Cargo.toml). Runs all gpu-gated parity tests for
# f32 / Q4 / fused attention against synthetic inputs.
test-cuda:
	@if [ "${LARQL_CUDA_AVAILABLE}" != "1" ]; then \
	  echo "Set LARQL_CUDA_AVAILABLE=1 to run CUDA parity tests."; \
	  exit 2; \
	fi
	cargo test -p larql-compute --features cuda --test test_cuda_f32 -- --test-threads=1
	cargo test -p larql-compute --features cuda --test test_cuda_q4  -- --test-threads=1
	cargo test -p larql-compute --features cuda --test test_cuda_attn -- --test-threads=1
	cargo test -p larql-rotorquant
	@echo "All CUDA + RotorQuant parity tests passed."

ci-cuda: test-cuda

# Snapshot of cuda backend status against this dev box.
cuda-status:
	@echo "═══ CUDA capability snapshot ═══"
	@echo
	@nvidia-smi 2>/dev/null | grep -E 'CUDA Version|GeForce|Tesla|RTX|GTX' | head -5 || echo "(no nvidia-smi)"
	@echo
	@if [ -d "/usr/local/cuda/targets/x86_64-linux/lib" ]; then \
	  echo "libcublas: $$(ls /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.* 2>/dev/null | head -1)"; \
	fi
	@echo
	@cargo metadata --format-version 1 --quiet 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); pkg=[p for p in d['packages'] if p['name']=='larql-compute'][0]; print(f\"larql-compute features: {list(pkg['features'].keys())}\")" 2>/dev/null || true

# ── Two-container deployment ──────────────────────────────────────────
# See deploy/docker/README.md for the full topology + VRAM budget.

docker-ffn:
	docker build -f deploy/docker/Dockerfile.ffn -t larql-ffn:dev .

docker-gpu:
	docker build -f deploy/docker/Dockerfile.gpu -t larql-gpu:dev .

docker-up:
	cd deploy/docker && docker compose up --build

docker-up-cpu:
	cd deploy/docker && docker compose -f docker-compose.cpu.yml up --build

docker-down:
	cd deploy/docker && docker compose down -v

docker-logs:
	cd deploy/docker && docker compose logs -f

# attention-service-routes change. Run a full HTTP smoke test against
# a running attention server (defaults to http://localhost:8081).
# Override target via: LARQL_ATTN_URL=... LARQL_MODEL_ID=... make attention-smoke
attention-smoke:
	python3 scripts/attention-service-smoke.py

# Numerical-validation harness for the attention-service routes —
# runs against the synthetic make_test_weights model in-process,
# bit-comparing every per-layer residual against a direct
# larql_inference forward pass. No network, no real model.
attention-validate:
	cargo test -p larql-server --test test_attention_validation

# Same as attention-validate, but at Gemma-3-4B-shaped synthetic
# dimensions (hidden=2560, num_q=8, num_kv=4, head_dim=320,
# 4 layers). Takes ~10s due to the synthetic weight build.
attention-validate-gemma:
	cargo test -p larql-server --test test_attention_validation -- --ignored prefill_gemma_shaped

# Latency benchmarks for the attention-service routes — drives
# prefill at seq_len ∈ {1, 8, 32, 128}, plus decode and snapshot
# after a 32-token prefill, against the synthetic model. Reports
# numbers under target/criterion/.
attention-bench:
	cargo bench -p larql-server --bench attention_service

# Check (compile without building)
check:
	cargo check --workspace

# Code quality
fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

lint:
	cargo clippy --workspace --tests -- -D warnings

# All quality checks. The traceability gate is fast (pure Python over markdown +
# `.rs` files) and runs in CI on every PR. The coverage gate is heavier; it has
# its own target (`ci-coverage`) and is wired into a separate CI job so spec-only
# PRs don't pay the cost.
ci: fmt-check lint test-full traceability-check openspec-validate

# Clean
clean:
	cargo clean

# Benchmarks
#
# `bench` runs the full quant_matvec suite and writes HTML reports under
# `target/criterion/`. `bench-save` records a baseline named `main`;
# `bench-check` re-runs and fails if any cell regresses past Criterion's
# default noise threshold. Plug `bench-check` into CI to catch the next
# 4× throughput cliff (the kind the q4_matvec_v4 row-drop bug caused) at
# PR time, not at goldens-fail time weeks later.
bench:
	cargo bench -p larql-compute --bench quant_matvec --features metal

bench-save:
	bash scripts/bench-regress.sh save

bench-check:
	bash scripts/bench-regress.sh check

# Demos
demos:
	cargo run --release -p larql-models --example architecture_demo
	cargo run --release -p larql-core --example graph_demo
	cargo run --release -p larql-core --example edge_demo
	cargo run --release -p larql-core --example serialization_demo
	cargo run --release -p larql-core --example algorithm_demo

demos-inference:
	cargo run --release -p larql-inference --example inference_demo

# Benchmarks
bench: bench-core

bench-core:
	cargo run --release -p larql-core --example bench_graph

bench-inference:
	cargo run --release -p larql-inference --example bench_inference

# Vindex micro-benches — synthetic, fast, safe under load.
bench-vindex:
	cargo bench -p larql-vindex --bench vindex_ops

# Vindex production-dim scaling bench. Refuses if larql-server / router
# are alive (they distort 1-2 GB matmuls). Run alone, on a cool host;
# results feed PERFORMANCE.md.
bench-vindex-scaling:
	@if pgrep -fl 'larql-(server|router)' >/dev/null 2>&1; then \
		echo "Refusing bench-vindex-scaling: larql daemons running. Stop them first."; \
		pgrep -fl 'larql-(server|router)'; \
		exit 2; \
	fi
	cargo bench -p larql-vindex --bench vindex_scaling

bench-all: bench-core bench-inference bench-vindex

# Coverage — uses cargo-llvm-cov.
#
#   coverage          — full workspace coverage; emits HTML + JSON.
#   coverage-summary  — terse per-crate text summary.
#   coverage-check    — enforce per-crate thresholds in coverage-thresholds.toml.
#   ci-coverage       — combined: regenerate + check; intended for CI.
#   coverage-install  — install rustup component + cargo-llvm-cov.
#
# Pass CRATE=<name> to scope `coverage` to a single crate (HTML only).
COVERAGE_CRATE ?=
coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Run: make coverage-install"; \
		exit 1; \
	fi
	@if [ -n "$(COVERAGE_CRATE)" ]; then \
		cargo llvm-cov --package $(COVERAGE_CRATE) --html --output-dir target/llvm-cov/html; \
		echo "Report: target/llvm-cov/html/index.html"; \
	else \
		cargo llvm-cov --workspace --json --output-path target/llvm-cov/coverage.json; \
		cargo llvm-cov --workspace --html --output-dir target/llvm-cov/html --no-clean; \
		echo "JSON:  target/llvm-cov/coverage.json"; \
		echo "HTML:  target/llvm-cov/html/index.html"; \
	fi

coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Run: make coverage-install"; \
		exit 1; \
	fi
	cargo llvm-cov --workspace --summary-only

coverage-check:
	@if [ ! -f target/llvm-cov/coverage.json ]; then \
		echo "target/llvm-cov/coverage.json missing — run \`make coverage\` first."; \
		exit 1; \
	fi
	python3 scripts/coverage-check.py

ci-coverage: coverage coverage-check

coverage-install:
	rustup component add llvm-tools-preview
	cargo install cargo-llvm-cov --locked

# OpenSpec spec → test traceability.
#
#   traceability         — regenerate openspec/coverage/traceability.{md,json}.
#   traceability-check   — fail if regenerated output diverges from committed.
#   gaps-unbacked        — write openspec/changes/<change>/gaps-unbacked-scenarios.md.
#   gaps-untested        — write gaps-untested-code.md (requires coverage JSON).
#   gaps                 — both gap reports.
#   openspec-validate    — `openspec validate <change> --strict` for the active change.
traceability:
	python3 scripts/spec-trace.py

traceability-check:
	python3 scripts/spec-trace.py --check

gaps-unbacked:
	python3 scripts/spec-trace.py --unbacked --quiet

gaps-untested:
	@if [ ! -f target/llvm-cov/coverage.json ]; then \
		echo "target/llvm-cov/coverage.json missing — run \`make coverage\` first."; \
		exit 1; \
	fi
	python3 scripts/spec-gap.py --untested-code

gaps: gaps-unbacked gaps-untested

openspec-validate:
	@for change in $$(ls openspec/changes 2>/dev/null | grep -v '^archive$$'); do \
		echo "openspec validate $$change --strict"; \
		openspec validate $$change --strict || exit 1; \
	done

# Python extension (managed via uv)
python-setup:
	cd crates/larql-python && uv sync --no-install-project --group dev

python-build: python-setup
	cd crates/larql-python && uv run --no-sync maturin develop --release

python-test: python-build
	cd crates/larql-python && uv run --no-sync pytest tests/ -v

python-check:
	cargo check -p larql-python

python-clean:
	rm -rf crates/larql-python/.venv crates/larql-python/uv.lock

# Extraction
extract-test:
	cargo run --release -p larql-cli -- weight-extract google/gemma-3-4b-it \
		--layer 26 -o output/test-L26.larql.json \
		--stats output/test-L26-stats.json

extract-full:
	cargo run --release -p larql-cli -- weight-extract google/gemma-3-4b-it \
		-o output/gemma-3-4b-knowledge.larql.json \
		--stats output/gemma-3-4b-stats.json

# Inference
predict:
	cargo run --release -p larql-cli -- predict google/gemma-3-4b-it \
		--prompt "The capital of France is" -k 10
