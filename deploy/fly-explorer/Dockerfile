# syntax=docker/dockerfile:1
# The public explorer image: larql-server in PUBLIC_EXPLORER mode over
# the generated demo container. See deploy/fly-explorer/README.md.
FROM rust:1-slim AS builder
WORKDIR /build

# The commit this image is built from, surfaced by the server as
# `server.revision` in GET /v1/capabilities so a deploy can be asserted
# rather than inferred from behaviour. Passed by the deploy:
#
#   fly deploy --build-arg LARQL_SERVER_REVISION=$(git rev-parse HEAD) ...
#
# Unset is a legitimate state: `option_env!` then yields None and the
# field is absent from the report. Never defaulted to a placeholder —
# a build that does not know its commit must not claim one.
ARG LARQL_SERVER_REVISION
ENV LARQL_SERVER_REVISION=${LARQL_SERVER_REVISION}

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
# larql-vindex embeds registry model records via include_str! at compile
# time — the build context must carry them.
COPY registry/ registry/

# curl + ca-certificates: utoipa-swagger-ui's build script downloads
# the Swagger UI bundle at compile time.
RUN apt-get update && apt-get install -y pkg-config libssl-dev protobuf-compiler cmake g++ libopenblas-dev curl ca-certificates && \
    cargo build --release -p larql-server --bin larql-server --bin vindex3-demo && \
    strip target/release/larql-server target/release/vindex3-demo

FROM ubuntu:24.04
RUN apt-get update && \
    apt-get install -y ca-certificates libssl3 curl libopenblas0 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/larql-server /usr/local/bin/
COPY --from=builder /build/target/release/vindex3-demo /usr/local/bin/
COPY deploy/fly-explorer/start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8080
CMD ["/start.sh"]
