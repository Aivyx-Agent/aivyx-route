# Changelog

## 0.2.0

Local-only. Breaking.

- Removed `EndpointKind::Anthropic` and `EndpointKind::Openai`; TOML naming
  them fails to parse (unknown variant).
- `locality = "local"` on an endpoint no longer makes a non-local address
  local: overrides can only lower trust (`"cloud"`). `validate` reports the
  no-op override as `LocalOverrideIgnored`.
- New `require_local` / `require_local_with` guard (`NotLocal`,
  `NotLocalReason`, `Resolve`, `SystemResolver`).
- New `ConfigIssue::NotLocal`, which products treat as fatal. `validate` may
  now do DNS lookups; `validate_with` takes the resolver.
- Discovery and residency check `require_local` before contacting an
  endpoint (the `discovery` feature now enables tokio's `rt` for
  `spawn_blocking`).
- `url_locality` is `Cloud` for any non-http(s) scheme with a hierarchical
  part (`unix:/x`, `foo:/evil.com`); `host:port` input is unchanged.
- `ConfigIssue::NonLocalAddress` now means "passes `require_local` only by
  DNS; its models count as cloud".
