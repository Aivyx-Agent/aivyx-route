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

On an Ollama endpoint, a tagless roster `id` (`llama3.2`) that discovery
reported only as `llama3.2:latest` enriches that discovered model, under
your name for it, instead of adding a second one. Listing both spellings
names one model (kept as `llama3.2:latest`); both entries apply to it, in
roster order.

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

### Stateful routing

`select` is pure and stateless; `Router` is the stateful layer both products
wrap around it. It tracks per-session stickiness, operator pins, failure
cooldowns, and each session's last decision, and stays synchronous and
I/O-free — callers inject the clock (`now: Instant`) instead of the router
reading it itself, so behavior is deterministic under test.

```rust
use aivyx_route::{Router, RouteQuery, TaskKind, TaskOverrides};
use std::time::Instant;

let router = Router::new(profiles, TaskOverrides::default());
let query = RouteQuery { session: Some("s".into()), ..RouteQuery::new(TaskKind::Chat) };
let plan = router.plan(&query, Instant::now())?;
// ... call plan.chain[0], falling back through the rest on a retryable error ...
let record = router.succeeded(&plan, &served_key, &failure_notes);
```

Semantics (operator-approved in model-routing Part 2):

- cooling models (a recent `failed()`) are a last resort, never a reason to
  fail a request;
- a fallback, or a choice made while any model was cooling, never becomes a
  session's sticky model;
- a pin never writes the sticky map, and `unpin` clears it too.

## Residency (Part 4)

`Policy.residency` is a `ResidencySnapshot`: which models are loaded, and
the host's VRAM. `select` scores it as a soft cost of at most one
tier-step — loaded models rank first, then loads that fit comfortably,
then loads that don't, and residency never affects hard filtering,
stickiness or pins. An empty snapshot (the default) changes nothing:
every model costs the same, so ordering is unchanged. `Router::set_residency`
replaces the snapshot the router's `plan` scores with; a product refreshes
it, it doesn't rebuild the router.

`discovery::residency::collect` (feature `discovery`) builds a snapshot
from up to three signals: Ollama's `/api/ps` + `/api/tags`, llama-server
router mode's `/models`, and an `aivyx-broker` reporting a product's
default backend, VRAM and slot pressure — the broker's VRAM wins when
present. `[routing] vram_bytes` is the operator's own VRAM figure, used as
the total only when nothing else reports one. `collect` is meant to be
polled on a short TTL (5s), never per request — like discovery, it's I/O,
so it stays behind the `discovery` feature.

Ollama reports a tagless model as `name:latest`; each such entry also
answers to the bare `name` (what `ollama run` accepts and operators often
configure), with its VRAM counted once.

A snapshot's `resident_endpoints` marks single-model servers: a model with
no entry of its own on such an endpoint counts as loaded, whatever the
product calls it (a per-model entry still wins). `collect` sets it for a
broker reporting exactly one loaded model; products mark an
OpenAI-compatible default backend resident themselves (Part 4b).

## Development

```sh
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo deny check licenses
```

Design: `docs/superpowers/specs/2026-09-25-model-routing-design.md`.
Licensed BUSL-1.1 — see `LICENSE` and `COMMERCIAL.md`.
