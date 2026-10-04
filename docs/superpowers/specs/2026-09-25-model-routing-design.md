# Task-Aware Model Routing (`aivyx-route`) Design

## Context

Neither `aivyx-pa` nor `aivyx-coder` can choose the right LLM for the task
at hand. Each runs every call against one configured model; the only
per-call variation today is a handful of static, hand-pinned overrides.
This spec designs task-aware model routing for the whole ecosystem, as a
new shared crate adopted by both products — the same "small shared crate,
adopted by both" pattern as `aivyx-confine` / `aivyx-checkpoint` /
`aivyx-kvcache` / `aivyx-skills`.

Operator-ranked concerns, in priority order (this ordering drives the part
sequencing below):

1. **B — Capability matching**: route by what a request *needs* (vision,
   tool calling, long context, audio, code/reasoning strength).
2. **A — Quality vs. speed tiering**: small/fast models for trivial work,
   large models for hard reasoning and multi-step coding.
3. **C — Local ↔ cloud escalation** (`aivyx-pa` only; `aivyx-coder` never
   calls a cloud API): escalate when local can't do it, with consent.
4. **D — Resource-awareness**: prefer what's loaded / fits in VRAM, avoid
   model-swap thrash on a single GPU.

Decisions confirmed with the operator during brainstorming (2026-09-25):

| # | Decision | Choice |
|---|---|---|
| 1 | What "correct" means | B, then A, then C, then D |
| 2 | Where the candidate/capability list comes from | **Both**: auto-discovery + operator roster; roster enriches/overrides |
| 3 | How task-side needs are classified | **Rules first, classifier optional**: deterministic requirement extraction + call-site `TaskKind` hints; optional small-model classifier, off by default |
| 4 | When `aivyx-pa` escalates to cloud | **All three triggers, each independently toggleable** (`no_local_candidate`, `on_failure`, `tiers`), one shared `never`/`ask`/`auto` consent gate |
| 5 | Private data vs. cloud | **Sensitive-source taint hard-blocks cloud escalation** for the whole conversation |
| 6 | Routing granularity | **Per call, sticky within a conversation**: main thread stays put unless a hard need forces a switch; side calls route freely |
| 7 | Architecture | **Shared crate `aivyx-route`**, thin per-product routing layers; `aivyx-broker` later becomes a residency-signal source for D |

Further decisions from the Part 1 final review (2026-09-26), folded into
the sections below:

| # | Decision | Choice |
|---|---|---|
| 8 | Capabilities a discovery source can't report | **Unknown, not absent**: an unknown capability passes a hard need but ranks below every model known to have it, and the reason says "assumed but unverified" |
| 9 | Endpoint names not declared anywhere | **Fail closed**: treated as `Cloud` + `Unavailable` (a typo must never make a cloud model look local); `RoutingConfig::validate()` reports it |
| 10 | Model identity | **`ModelKey` = (endpoint, id)**: the same id on two endpoints is two models; `Policy` stickiness/exclusion use keys, not bare ids |
| 11 | VRAM source | No LLM backend reports free/total VRAM, so **`aivyx-broker` reads the GPU itself** (`nvidia-smi`, else the AMD card with the most VRAM via sysfs); `[routing] vram_bytes` is the operator fallback without a broker; with neither, no won't-fit term |
| 12 | Residency weight | Residency cost in **quarter tier-steps** added to the tier penalty (one ranking column): loaded 0, no signal 2, needs a load 1–3 by size, won't fit 4 (= one tier-step, never disqualifying) |
| 13 | Part 4 split | **4a**: crate scoring + signal readers + broker endpoint; **4b**: periodic refresh wired into both products |

## Grounding

Verified directly against the code and upstream docs before writing this
spec, not assumed:

- **`aivyx-coder` today**: one `[backend]` (`base_url` + `model`), plus
  separately-configured, manually-invoked endpoints for `/architect`
  (`ArchitectSettings`), council (`CouncilMember`), and verification.
  Team specialists all share the main backend (no per-role model).
  `LlmBackend` (`crates/aivyx-llm/src/backend.rs`) binds one model per
  backend instance (`fn model_id()`); `ChatRequest` carries **no** model
  field.
- **`aivyx-coder` has no image input**: `aivyx-types::ContentBlock` is
  `Text | ToolCall | ToolResult` only. A vision requirement therefore can't
  arise in `aivyx-coder` today; the crate supports it, `aivyx-coder` simply
  never triggers it until image input exists.
- **`aivyx-coder`'s compaction makes no LLM call**: `compact_if_needed`
  (`crates/aivyx-core/src/agent/mod.rs`) drops oldest turns. There is no
  `Compact` side call to route in `aivyx-coder`.
- **`aivyx-pa` today**: one `provider` (`ProviderKind`: Anthropic, OpenAi,
  Ollama, LlamaCpp, Jan, MistralRs, Broker) + one `model`. Static overrides:
  `judge_model: Option<String>` (config `lib.rs`), Chapter Ensemble's
  per-role `model` + endpoint (`aivyx-team-types/src/config.rs`). Chapter
  Foreman's `delegate_above` is the only task-aware logic — a structural
  complexity score that picks *solo vs. team*, never a model.
  `LlmRequest` (`crates/aivyx-llm/src/lib.rs`) **does** carry `model: &str`
  per request; a provider instance serves one `ProviderKind`.
- **`aivyx-pa` has no general context-provenance tracking** (corrects an
  earlier brainstorming claim). `MessageOrigin` is only `Operator | System`.
  What does exist and is sufficient to build on: every prior tool call
  persists its `tool_name` in history (`LlmToolCallRecord`), and memory
  recall is injected via the `ContextProvider` trait
  (`aivyx-core/src/llm_planner.rs`).
- **Ollama discovery** (upstream docs, via context7): `/api/show` returns
  `capabilities: [...]` drawn from `completion, tools, insert, vision,
  embedding, thinking, image, audio`, plus `model_info.<arch>.context_length`.
  `/api/ps` lists loaded models with `size_vram`.
- **llama.cpp discovery** (upstream docs): `llama-server` **router mode**
  serves many models with dynamic load/unload; its `GET /models` returns
  per-model `id`, `status.value` (e.g. `loaded`) and
  `architecture.input_modalities` (e.g. `["text","image"]`), plus
  `POST /models/unload`. Classic single-model mode exposes only
  `/v1/models` ids.
- **No discovery source reports quality** (coding/reasoning skill). Tiers
  and strengths must come from the roster.
- **Name is free**: no local `aivyx-route/` directory, no
  `Aivyx-Agent/aivyx-route` GitHub repo.

## Decomposition

One spec, four parts, built in order. Each part gets its own
implementation plan.

| Part | Repo | Delivers | Concerns |
|---|---|---|---|
| **1** | new `aivyx-route` | core types, merge rules, requirement extraction, pure `select()`, discovery clients (feature-gated), classifier prompt/parse helpers | B, A |
| **2** | `aivyx-coder` | `RoutedBackend` + backend pool, `[routing]` config, stickiness, call-site tagging, `/models` + `/model` | B, A |
| **3** | `aivyx-pa` | `RoutedProvider`, `[routing]` config, stickiness, call-site tagging, cloud escalation, sensitivity taint, optional classifier wiring | B, A, C |
| **4** | `aivyx-broker` + `aivyx-route` | residency endpoint on the broker; residency scoring in the crate; product wiring (4a crate + broker, 4b products) | D |

The classifier is not its own part: the crate ships its prompt + strict
parser (Part 1); `aivyx-pa` wires it (Part 3). `aivyx-coder` never uses it.

## Part 1 — The `aivyx-route` crate

### Core types

- **`ModelProfile`** — one candidate:
  - `id: String`, `endpoint: EndpointRef`, `locality: Local | Cloud`;
    identified by `ModelKey { endpoint, id }` (`Display` as `id@endpoint`,
    via `ModelProfile::key()`)
  - `capabilities: CapabilitySet` over `Completion | Tools | Vision | Thinking | Audio | Embedding`
  - `unknown_capabilities: CapabilitySet` — capabilities whose presence
    discovery couldn't determine (empty for explicitly-built profiles)
  - `context_window: Option<u32>`
  - `tier: Small | Medium | Large`
  - `strengths: Set<Code | Reasoning | Chat | Summarize>`
  - `priority: i32` (operator tie-break; higher wins)
  - `availability: Available | Unverified | Unavailable { until }`
  - per-field provenance (`Discovered | Roster`) for `explain` output.
- **`Requirements`** — `hard: { vision, tools, audio, embedding: bool, min_context: Option<u32> }`
  (filters) and `soft: { tier: Option<Tier>, strengths: Set<Strength> }`
  (scoring).
- **`TaskKind`** — `Chat | CodeEdit | Plan | Judge | Summarize | Compact |
  Classify | Embed | Custom(String)`. Each maps to a default `(tier,
  strengths)`; the operator can override via `[routing.tasks]`. Defaults:

  | TaskKind | Tier | Strengths |
  |---|---|---|
  | `Chat` | Medium | Chat |
  | `CodeEdit` | Large | Code |
  | `Plan` | Large | Reasoning |
  | `Judge` | Medium | Reasoning |
  | `Summarize` | Small | Summarize |
  | `Compact` | Small | Summarize |
  | `Classify` | Small | — |
  | `Embed` | — (requires `Embedding` capability) | — |
  | `Custom(_)` | Medium | — (unless overridden) |

- **`Policy`** — `sticky_model: Option<ModelKey>` (the conversation's
  current model, for stickiness), `allow_cloud: bool`,
  `exclude: Vec<ModelKey>`. The crate enforces; each product decides the
  values.
- **`select(&Requirements, &[ModelProfile], &Policy) -> Result<Decision, NoCandidate>`**
  — pure and deterministic:
  1. Drop `Unavailable`, `exclude`d, and (unless `allow_cloud`) `Cloud`
     profiles.
  2. Drop every profile failing any hard requirement. A capability need is
     met if the capability is present **or unknown**. A profile whose
     `context_window` is unknown (neither discovered nor in the roster)
     *passes* the `min_context` filter. Both kinds of "passed on unknown"
     rank lower in step 4.
  3. If `sticky_model` survives step 2, return it (reason: "sticky").
  4. Otherwise score survivors: first, profiles meeting a capability need
     only via `unknown_capabilities` rank below every profile known to
     have it; then unknown-context-window profiles below known, sufficient
     ones; then tier fit (exact = best; one step off = penalty;
     larger-than-needed penalised less than smaller-than-needed), strength
     overlap, then `priority`, then endpoint and `id` lexical order as the
     final deterministic tie-break. (Part 4 adds a residency cost to the tier column — see Part 4.)
  5. Return `Decision { model, fallbacks (rest, ranked), reason }`.
- **`Decision.reason`** — structured (`Vec<ReasonPart>`) with a `Display`
  impl producing one human sentence, e.g. *"vision required;
  `llava:13b` is the only vision-capable local model."* A choice that
  relies on unknown capabilities adds *"assumed but unverified: tools"*;
  the unknown-context-window note appears only when the model was chosen
  by ranking, not kept by stickiness.
- **`NoCandidate`** — carries the list of unmet hard requirements (and,
  per requirement, the closest near-miss) so errors can say exactly what's
  missing. With zero candidates at all, every hard need is still listed
  (no near-miss) and the message reads *"no candidate models are
  available (needed: vision, tool calling)"*. `aivyx-pa` uses it as
  escalation trigger 1.
- **Consumer helpers** — `unmet_needs(&Requirements, &ModelProfile)` (the
  needs a profile definitely fails, same unknown-passes rule as `select`)
  and `find(&[ModelProfile], &ModelKey)`, used to validate explicit
  operator pins. Config and output types are `Serialize` (both products
  write config back / emit decisions as JSON), and `TaskKind` round-trips
  as its string name (`FromStr` never yields `Custom` for a built-in name).

### Requirement extraction

The crate does not know either product's message types. It exposes
`Requirements::builder()` with `.vision()`, `.tools()`, `.audio()`,
`.min_context(n)`, `.task(TaskKind)` (`.task(Embed)` also sets the hard
`embedding` requirement); each product writes a small
`fn requirements_for(&Request) -> Requirements` adapter against its own
types:

- `vision` ⇐ any image content block in the messages
- `tools` ⇐ non-empty tool list
- `min_context` ⇐ estimated prompt tokens + `max_tokens` (each product
  reuses its existing token estimator)
- `audio` ⇐ an audio content block (`aivyx-pa` voice only)

### Discovery (`discovery` cargo feature, default off)

Uses `reqwest` (already a dependency of both consumers). One client per
backend family, each returning partial `ModelProfile`s:

| Backend | Endpoint(s) | Yields |
|---|---|---|
| Backend | Endpoint(s) | Yields | Unknown capabilities |
|---|---|---|---|
| Ollama | `/api/tags`, then `/api/show` per model | ids, `capabilities[]`, `context_length` | none (all six if `/api/show` fails) |
| llama.cpp router mode | `GET /models` | ids, `input_modalities` (image ⇒ `Vision`), load status | completion, tools, thinking, embedding |
| llama.cpp single mode / Jan / generic OpenAI-compat | `GET /v1/models` | ids only | all six |
| Cloud (Anthropic / OpenAI) | none | roster only — never probed | all six until the roster declares them |

Ollama's `context_length` is the model's *trained* length, not the served
`num_ctx`; operators should set roster `context_window` to the serving
value (Part 4 reads the served value from `/api/ps`). The discovery module
re-exports `reqwest` so consumers on a different `reqwest` major (aivyx-pa
is on 0.12) can build a compatible client.

Discovery runs at startup and on explicit refresh only — never per call.
Results are cached with a timestamp. An unreachable endpoint marks its
models `Unavailable`; discovery failure is never fatal.

### Roster config (identical TOML shape in both products)

```toml
[routing]
enabled = true            # false or absent ⇒ exactly today's behavior
discover = true

[routing.endpoints.ollama-main]
kind = "ollama"
base_url = "http://localhost:11434"

[[routing.models]]
id = "qwen3-coder:30b"
endpoint = "ollama-main"  # omitted ⇒ the product's default backend
tier = "large"
strengths = ["code", "reasoning"]
priority = 10

[[routing.models]]
id = "llava:13b"
tier = "small"
capabilities = ["vision"]        # adds to discovery
capabilities_deny = ["tools"]    # removes a discovered-but-unreliable capability

[routing.tasks]
summarize = { tier = "small" }
```

### Merge rules

- Capabilities = discovered ∪ roster `capabilities`, minus roster
  `capabilities_deny`.
- `context_window`: roster value if set, else discovered.
- `tier`, `strengths`, `priority`: roster only. A discovered model with no
  roster entry defaults to `Medium`, no strengths, priority `0` — usable,
  still hard-filtered.
- A roster model discovery didn't find is kept, marked `Unverified` (keeps
  cloud models and single-mode llama.cpp working); if chosen and it fails,
  the fallback chain handles it.
- Unknown capabilities: from discovery (see the Discovery table); a
  roster-only model starts with all six unknown. Every roster
  `capabilities` and `capabilities_deny` entry makes that capability
  known (present or absent). `capabilities_deny` removes from discovered
  **and** declared capabilities; a deny from any duplicate entry wins.
- Duplicate roster entries for the same (endpoint, id) apply in order,
  each overriding only the fields it sets.
- Endpoint locality: `merge` takes the product's `DefaultEndpoint { name,
  kind, base_url, locality }`. The default name and every
  `[routing.endpoints]` key resolve by one rule
  (`effective_locality()`): cloud kinds are `Cloud` (so an aivyx-pa
  default of Anthropic is `Cloud`); other kinds follow their `base_url`
  host — local addresses only are `Local`, no address is `Cloud` — unless
  `locality` is set; **any other name fails
  closed** — `Cloud` and `Unavailable`, never selected. A roster
  `locality` only moves a model towards `Cloud` (never local on a cloud
  endpoint), and never changes availability. *(Amended 2026-10-04:
  originally the kind alone decided, and a roster `locality` overrode in
  either direction.)*
- `RoutingConfig::validate(&DefaultEndpoint) -> Vec<ConfigIssue>` (pure,
  deterministic order) reports `UnknownEndpoint`, `DuplicateModel`,
  `MissingBaseUrl` (a local kind with no `base_url`), `NonLocalAddress`
  and `LocalOverrideIgnored` so products can warn at startup.

### Classifier helpers

`classifier::prompt(&[message summaries]) -> String` and
`classifier::parse(&str) -> Option<Tier>` (accepts exactly
`small|medium|large`, case-insensitive, whitespace-trimmed; anything else
⇒ `None`). No I/O in the crate — the product makes the call.

### Licensing / packaging

`Aivyx-Agent/aivyx-route`, BUSL-1.1 from day one, `publish = false`,
legal files copied from `aivyx-ecosystem/docs/legal/` (same as the Part 2
relicense repos), `deny.toml` license gate matching the org-wide
allow-list. Consumers pin it by git `rev`, like every other shared crate —
so each consumer part must start by pushing the crate commit it pins.

## Parts 2 & 3 — Shared integration shape

### Plugging in

- **`aivyx-coder`**: `LlmBackend` binds one model per instance, so routing
  needs a **backend pool** keyed by `(endpoint, model)`, built lazily. A new
  `RoutedBackend` implements `LlmBackend` itself, so `Agent`, team, ACP and
  the MCP server are untouched. Its `model_id()` returns the model of the
  most recent decision (for the status line).
- **`aivyx-pa`**: `LlmRequest.model` is already per-request, so a
  same-provider switch rewrites that string. A cross-provider-kind switch
  (local → cloud) dispatches to a second provider instance. A new
  `RoutedProvider` implements `LlmProvider`; the planner is untouched.
  `tool_call_family_hint` forwards to the provider serving the chosen model.

### Tagging

Add an optional `task: Option<TaskKind>` to `ChatRequest` (`aivyx-coder`)
and `LlmRequest` (`aivyx-pa`), defaulting to `Chat`. Untagged call sites
keep today's behavior.

### Stickiness

The routing layer keeps a `session → current ModelKey` map (`aivyx-coder`:
session; `aivyx-pa`: conversation/channel thread). `Chat` and `CodeEdit`
calls pass it as `Policy.sticky_model`; if it fails a hard requirement,
`select()` re-picks, the map updates (new model sticks), and the reason is
surfaced. Side-call kinds (`Judge`, `Summarize`, `Compact`, `Classify`,
`Embed`, `Plan`) neither read nor write the map.

### First-pass call sites

| `aivyx-coder` | `aivyx-pa` |
|---|---|
| main turn loop → `CodeEdit` | planner turn → `Chat` |
| `/architect` → `Plan` (existing `[architect]` becomes an implicit roster entry; old section still honored) | judge → `Judge` (replaces `judge_model` resolution when routing is on) |
| verification → `Judge` | compaction / summarization → `Summarize` |
| team specialists → per-role `task` in the roster TOML | Foreman: when a story scores ≥ `delegate_above`, the delegated team lead's turns are tagged `Plan` (solo turns stay `Chat`) |
| council members stay explicit (a council is deliberately multi-model) | Ensemble per-role `model`, if set, pins that role |

### Explicit pins win

`judge_model`, Ensemble per-role `model`, and `aivyx-coder`'s
`[architect]`/council endpoints keep working unchanged. When set, they pin
that call: routing only validates hard requirements against the pinned
model and logs a warning on a mismatch — it never silently overrides an
explicit operator choice. `aivyx-coder`'s `/model <id>` pins the session
the same way.

### Visibility

- `aivyx-coder`: status-line model indicator; `/models` (list, `refresh`,
  `why` = last decision's reason); `/model <id>` session pin.
- `aivyx-pa`: `routing.status` / `routing.explain` tools and matching CLI
  subcommands.
- Every decision emits a `tracing` event with its reason; in `aivyx-pa` it
  is also an audit-chain entry.

## Part 3 only — Cloud escalation and sensitivity (`aivyx-pa`)

### Cloud candidates

Roster entries with `locality = "cloud"` and `endpoint = "anthropic"` /
`"openai"`, credentials via `aivyx-pa`'s existing key handling. Cloud
profiles enter the candidate set only when an escalation trigger fires and
the consent gate passes — ordinary selection runs with
`Policy.allow_cloud = false`.

### Triggers (evaluated in this order, each toggleable)

1. **`no_local_candidate`** (default **on**) — local `select()` returned
   `NoCandidate`.
2. **`on_failure`** (default **off**) — the local attempt demonstrably
   failed: tool-call output unparseable after existing retries, a Verdict
   judge `FAIL`, or a Circuit stall-breaker trip. Escalates **the retry
   only**; the conversation's sticky model stays local.
3. **`tiers`** (default **empty**) — listed task kinds / tiers route
   straight to cloud, e.g. `tiers = ["plan"]`.

### Consent gate

`escalation.mode = "never" | "ask" | "auto"`, default **`ask`**, shared by
all triggers. `ask` uses the existing per-channel approval flow; the prompt
names the model, the trigger, and an approximate outbound token count, and
offers "allow for this conversation". `never` means cloud profiles are
never candidates.

### Sensitivity taint (overrides everything)

A conversation becomes **tainted** when any of:

- a tool whose name matches a sensitive prefix produced output in it —
  default prefixes `gmail.`, `calendar.`, `contacts.`, `drive.`, `memory.`,
  `notion.`, `obsidian.`, `fs.read`; configurable in `[routing.sensitive]`
- a `ContextProvider` reporting `sensitive() == true` injected content —
  new trait method `fn sensitive(&self) -> bool { false }`; memory recall
  overrides to `true`
- the inbound message arrived on a channel marked sensitive (email-backed
  channels by default)

Taint is **persisted on the conversation and sticky for its lifetime**. It
is deliberately *not* re-derived from current history, because compaction
would launder it (a summary can contain Gmail content while no longer
containing the Gmail tool call). A tainted conversation makes every
trigger fall back to the best local model — or a clear error if none —
with the reason surfaced ("cloud escalation blocked: conversation contains
gmail output") and audited. `auto` does not bypass taint; there is no
per-conversation override in v1 (deliberate; can be added later if
wanted).

### Audit

Every escalation — allowed, denied, or taint-blocked — writes an audit
entry: model, trigger, mode, consent outcome, and a hash of the outbound
payload. Never the content.

### Classifier wiring

`[routing.classifier] enabled = false` by default; picks its model via
`TaskKind::Classify`. Runs only when a call is `TaskKind::Chat` **and** no
rule set a tier (free-form chat). Strict grammar output via the existing
`tool_grammar` module, 2s timeout, `None` ⇒ `Medium`.

## Part 4 — Resource-awareness

Split into **4a** (`aivyx-route` + `aivyx-broker`) and **4b** (product
wiring) — decision 13.

- **Crate — data**: `ResidencySnapshot { models: BTreeMap<ModelKey,
  ModelResidency>, resident_endpoints: BTreeSet<EndpointRef>, vram:
  Option<Vram>, slots: BTreeMap<EndpointRef, SlotPressure> }` with
  `ModelResidency::{Loaded { vram_bytes }, NotLoaded { size_bytes }}`
  (both `Option<u64>`). I/O-free. Passed to `select()` as
  `Policy.residency`; `Router::set_residency` stores the latest one for
  `plan()`.
- **Resident endpoints** (amended 2026-09-26, final review I1): a model
  with no entry of its own on an endpoint in `resident_endpoints` counts
  as `Loaded` (cost 0). This is for single-model servers, whose one
  model is resident whatever the product calls it — the server's own id
  (a llama-server `--alias` or model path) rarely equals the product's
  configured model name, so a per-id entry would never match. A per-model
  entry always wins over the endpoint signal. Resident endpoints add no
  evictable VRAM (their size is unknown). Without this, "no entry" (2)
  would lose to a small cold load (1) and routing would force a load
  beside an always-loaded default backend.
- **Crate — scoring** (decision 12): `rank_key`'s tier column becomes
  `tier_penalty × 4 + residency_cost`. Cost: no entry 2; `Loaded` 0;
  `NotLoaded` with known size and available VRAM: > available ⇒ 4,
  ≤ ¼ ⇒ 1, ≤ ½ ⇒ 2, else 3; otherwise 2. *Available* = total − (used −
  VRAM held by the snapshot's loaded models), i.e. what a load could use
  after evicting every loaded model. Residency is **soft only**: it never
  affects hard filtering, stickiness or pins, and it is worth at most one
  tier-step, ranked ahead of strengths and priority exactly as tier fit
  is. An empty snapshot leaves every decision unchanged. A chosen model's
  residency adds a reason clause ("already loaded" / "needs loading" /
  "may not fit in free VRAM").
- **Signal sources** (crate, `discovery` feature, `discovery::residency`):
  Ollama `/api/ps` (loaded, `size_vram`) + `/api/tags` (`size`);
  llama.cpp router `GET /models` `status.value` (`loaded`/`loading` ⇒
  loaded, `unloaded`/`sleeping` ⇒ not loaded, no `status` ⇒ loaded —
  single-model server); the broker (below). Brokers front a product's
  *default* backend (never a `[routing.endpoints]` entry), so broker
  residency is keyed to the default endpoint. A broker reporting exactly
  one model, and that model loaded, also puts the default endpoint in
  `resident_endpoints`. In 4b, products mark a single-model
  OpenAI-compatible default backend resident themselves (nothing to ask
  it). An unreachable source contributes nothing; `collect` never fails.
- **VRAM** (decision 11): the broker reads the GPU (`nvidia-smi`, summed
  across GPUs; else the AMD card with the largest
  `mem_info_vram_total`; `--vram-source auto|nvidia|amd|none`). Without a
  broker VRAM figure, `[routing] vram_bytes` supplies the total and used
  = VRAM held by loaded models. One host-wide pool is assumed for all
  local endpoints (multi-host is out of scope).
- **Broker endpoint**: read-only `GET /v1/aivyx/residency` (loopback, no
  auth, like the rest of the broker):
  ```json
  {"models":[{"id":"qwen3-8b","loaded":true}],
   "vram":{"total_bytes":25769803776,"used_bytes":9663676416},
   "slots":{"busy":1,"total":2}}
  ```
  `models` from the upstream `GET /models`; `vram` is `null` when
  unknown; `slots` from the broker's scheduler. Slot pressure is reported
  and parsed but not scored in 4a.
- **Refresh (4b)**: products poll `collect` on a short TTL (default 5s)
  in the background and hand the result to `Router::set_residency` —
  never per call.

## Error handling

- **Chosen model fails** (connection error, 404 model-not-found, load
  timeout): mark it `Unavailable` for a cooldown (default 60s) and try the
  next `Decision.fallbacks` entry; the reason records the chain tried.
- **Fallbacks exhausted**: return the original error plus the attempted
  chain. In `aivyx-pa` this is where `on_failure` escalation may apply.
- **Discovery failure**: never fatal (see Discovery).
- **`NoCandidate` with escalation off/blocked**: an actionable error naming
  the unmet requirement, e.g. *"this request contains an image; no local
  model has vision — add one to `[routing.models]` or pull `llava`."*
  Never a silent dispatch to an incapable model.
- **Mid-stream failure**: no retry once tokens have streamed (avoids
  duplicated partial output); surfaced exactly as today.

## Compatibility

With `[routing]` absent or `enabled = false`, both products behave
exactly as today — no config migration, and every existing test passes
unchanged. This is itself a tested invariant in Parts 2 and 3.

## Testing

- **Crate**: table-driven unit tests for `select()` (each hard filter,
  stickiness, tie-break determinism, tier-mapping, `NoCandidate`
  near-misses); unit tests for each merge rule; discovery clients tested
  against recorded real JSON fixtures (Ollama `/api/show` + `/api/ps`,
  llama.cpp `/models`); classifier `parse` accepts/rejects table.
- **Products**: routing-off ⇒ byte-identical behavior; stickiness (a
  forced switch sticks; side calls don't disturb it); explicit pins win.
- **`aivyx-pa`**: taint survives compaction; every trigger × every mode ×
  tainted/untainted matrix; audit entries written with hash, never
  content.
- **Control experiments** (org precedent — it has caught a Critical
  vacuous-test bug before): temporarily disable the taint gate and confirm
  the taint tests fail; likewise disable the hard-requirement filter and
  confirm the `NoCandidate` tests fail. Revert.
- **Live verification**: one end-to-end run per product against a real
  Ollama serving two models (one vision-capable, for `aivyx-pa`).

## Out of scope (v1)

- Learned/benchmark-driven quality scores (tiers/strengths are
  operator-declared).
- A per-conversation override of sensitivity taint.
- Cloud routing in `aivyx-coder` (it is local-only by design).
- Image input for `aivyx-coder` (would make its vision routing live; a
  separate feature).
- Automatic model pulling/downloading.
