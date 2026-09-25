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
   Ranking: known capabilities first, then known-sufficient context, then
   tier fit, strength overlap, operator `priority`, and finally endpoint and
   model id.
3. **Stickiness.** The product passes the conversation's current model; it
   is kept whenever it still meets every hard need.

A model is identified by its `ModelKey` — `(endpoint, id)` — so the same id
served by two endpoints is two models.

### Unknown capabilities

Some sources can't say what a model supports: `/v1/models` on an
OpenAI-compatible server reports ids only, llama-server router mode reports
only input modalities (so vision/audio are known, the rest aren't), and a
roster-only model has nothing discovered. Those capabilities are *unknown*,
not absent: an unknown capability passes a hard need, but the model ranks
below every model **known** to have it, and the decision says
*"assumed but unverified: tools"*. Declare `capabilities` or
`capabilities_deny` in the roster to make a capability known either way.

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
capabilities_deny = ["tools"]    # removed from discovered and declared

[routing.tasks]
summarize = { tier = "small" }
```

Duplicate roster entries for the same `(endpoint, id)` apply in order, each
overriding only the fields it sets; a capability denied by any of them is
always removed.

An `endpoint` that is neither the product's default nor a
`[routing.endpoints]` key (a typo, say) fails closed: its models are treated
as cloud and unavailable, so they are never selected.
`RoutingConfig::validate(&default)` reports that, duplicate models, and
local endpoints with no `base_url`; products warn at startup.

Discovery (cargo feature `discovery`) reads Ollama's `/api/tags` +
`/api/show`, llama-server router mode's `/models`, and `/v1/models` on any
OpenAI-compatible server. Cloud endpoints are never probed. Tiers and
strengths always come from the roster — no backend reports quality.
Ollama's discovered context window is the model's *trained* length, not
the `num_ctx` it is served with: set the roster `context_window` to your
serving value (Part 4 will read it from `/api/ps`). The discovery module
re-exports `reqwest`, so a consumer on another major builds its client as
`aivyx_route::discovery::reqwest::Client::new()`.

## Use

```rust
use aivyx_route::{
    DefaultEndpoint, EndpointKind, EndpointRef, Policy, Requirements, TaskKind, merge, select,
};

let default = DefaultEndpoint { name: EndpointRef::new("main"), kind: EndpointKind::Ollama };
for issue in config.validate(&default) {
    eprintln!("warning: {issue}");
}
let profiles = merge(&config, &default, &reports);
let req = Requirements::builder()
    .task(&TaskKind::CodeEdit, &config.tasks)
    .tools()
    .build();
let decision = select(&req, &profiles, &Policy::default())?;
println!("{decision}"); // chose `qwen3-coder:30b`: tool calling required; ...

// Next turn: keep the conversation on the same model if it still qualifies.
let policy = Policy { sticky_model: Some(decision.model.key()), ..Policy::default() };
```

`unmet_needs(&req, &profile)` and `find(&profiles, &key)` validate an
explicit operator pin; config and decision types are `Serialize`.

## Development

```sh
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo deny check licenses
```

Design: `docs/superpowers/specs/2026-09-25-model-routing-design.md`.
Licensed BUSL-1.1 — see `LICENSE` and `COMMERCIAL.md`.
