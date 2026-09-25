# aivyx-route

[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-blue.svg)](LICENSE)

Task-aware model routing shared by `aivyx-pa` and `aivyx-coder`. Given the
models you have (auto-discovered and/or declared) and what a request needs,
it picks one deterministically and says why.

## How it decides

1. **Hard needs filter.** Vision, tool calling, audio, embeddings, minimum
   context window. A model failing any is never chosen.
2. **Soft preferences rank.** The call site's task kind (`chat`,
   `code_edit`, `plan`, `judge`, `summarize`, `compact`, `classify`,
   `embed`, or a custom name) maps to a preferred tier and strengths.
   Ranking: known-sufficient context first, then tier fit, strength
   overlap, operator `priority`, and finally model id.
3. **Stickiness.** The product passes the conversation's current model; it
   is kept whenever it still meets every hard need.

## Configuration

Both products embed the same `[routing]` section:

```toml
[routing]
enabled = true            # false or absent ⇒ today's single-model behavior
discover = true

[routing.endpoints.ollama-main]
kind = "ollama"           # ollama | llama_router | openai_compat | anthropic | openai
base_url = "http://localhost:11434"

[[routing.models]]
id = "qwen3-coder:30b"
endpoint = "ollama-main"  # omitted ⇒ the product's default backend
tier = "large"            # small | medium | large
strengths = ["code", "reasoning"]
priority = 10

[[routing.models]]
id = "llava:13b"
tier = "small"
capabilities = ["vision"]        # added to what discovery found
capabilities_deny = ["tools"]    # removed from what discovery found

[routing.tasks]
summarize = { tier = "small" }
```

Duplicate roster entries for the same `(endpoint, id)` apply in order, but a
capability denied by any of them is always removed.

Discovery (cargo feature `discovery`) reads Ollama's `/api/tags` +
`/api/show`, llama-server router mode's `/models`, and `/v1/models` on any
OpenAI-compatible server. Cloud endpoints are never probed. Tiers and
strengths always come from the roster — no backend reports quality.

## Use

```rust
use aivyx_route::{Policy, Requirements, TaskKind, merge, select};

let profiles = merge(&config, &default_endpoint, &reports);
let req = Requirements::builder()
    .task(&TaskKind::CodeEdit, &config.tasks)
    .tools()
    .build();
let decision = select(&req, &profiles, &Policy::default())?;
println!("{decision}"); // chose `qwen3-coder:30b`: tool calling required; ...
```

## Development

```sh
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo deny check licenses
```

Design: `docs/superpowers/specs/2026-09-25-model-routing-design.md`.
Licensed BUSL-1.1 — see `LICENSE` and `COMMERCIAL.md`.
