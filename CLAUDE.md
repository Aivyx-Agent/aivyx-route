# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## What this is

`aivyx-route` is task-aware model routing shared by `aivyx-pa` and
`aivyx-coder`: given candidate models (auto-discovered from backends and/or
declared in an operator roster) and what a request needs (vision, tools,
context size, task kind), it deterministically picks a model and explains
why. Same "small shared crate, adopted by both" pattern as
`aivyx-confine`/`aivyx-checkpoint`/`aivyx-kvcache`/`aivyx-skills`.

Design: `docs/superpowers/specs/2026-09-25-model-routing-design.md` (this
crate is Part 1; Parts 2–4 live in `aivyx-coder`, `aivyx-pa`,
`aivyx-broker`).

## Build, test, lint

```sh
cargo build --all-features
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo deny check licenses
```

Single crate, no workspace — no `-p` flag needed. Plain `cargo test`
skips the discovery clients (feature `discovery`, off by default).

## Architecture

- `src/profile.rs` — `ModelProfile` and its vocabulary types.
- `src/requirements.rs` — `Requirements`, `TaskKind`, task→tier defaults.
- `src/select.rs` — the pure, deterministic `select()`.
- `src/config.rs` — the `[routing]` TOML shape both products embed.
- `src/merge.rs` — discovery reports + roster → `Vec<ModelProfile>`.
- `src/classifier.rs` — optional small-model classifier prompt/parse.
- `src/discovery/` — HTTP clients (feature `discovery`); the only I/O.

Models are identified by `ModelKey` (endpoint, id), never by id alone.

Everything outside `src/discovery/` must stay I/O-free and synchronous.
