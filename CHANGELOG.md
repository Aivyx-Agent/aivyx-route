# Changelog

## 0.2.0

Local-only. Breaking.

- Removed `EndpointKind::Anthropic` and `EndpointKind::Openai`; TOML naming
  them fails to parse (unknown variant).
- `locality = "local"` on an endpoint is honoured only for an address that
  isn't local by spelling and only after `RoutingConfig::verify_local`
  (products must call it at load) confirmed `require_local` passes for it;
  that records the verified URL in a new private, non-TOML `dns_verified`
  field on `EndpointConfig` and `DefaultEndpoint` (readable as
  `dns_verified()`; changing `base_url` afterwards makes it stale, so it
  fails closed; excluded from `==`). **Struct literals of `EndpointConfig` /
  `DefaultEndpoint` no longer compile outside the crate (E0451)**: use the new
  `EndpointConfig::new(kind, base_url)` + `.with_locality(..)` and
  `DefaultEndpoint::new(name, kind)` + `.with_base_url(..)` /
  `.with_locality(..)`. `verify_local(&mut default)` uses the system
  resolver; `verify_local_with(&mut default, &dyn Resolve)` takes one. `effective_locality`: `cloud` override, else a local
  address, else verified `local` override, else Cloud (unverified fails
  closed). `LocalOverrideIgnored` (endpoint) now means "override set but not
  verified"; `NonLocalAddress` means "resolves all-local, no override: set
  `locality = \"local\"`". `verify_local` reports `NotLocal` (fatal) for an
  override whose URL fails the guard.
- Connect-time resolver: `resolve_local` / `resolve_local_with(host, port)`
  (reqwest-independent: all addresses must be local; cloud hosts refused
  without a lookup) and, with the `discovery` feature, `LocalOnlyDns`, a
  `reqwest::dns::Resolve` built on it (runs in `spawn_blocking`; the
  `NotLocal` is the error). `discovery::client()` installs it. Recommended
  client for products: `Client::builder().dns_resolver(Arc::new(LocalOnlyDns))
  .redirect(Policy::none())`; connections to a name can then only go to local
  addresses whatever DNS does after the check. reqwest does not call the
  resolver for IP-literal URLs, so keep using `require_local` too.
- New `require_local` / `require_local_with` guard (`NotLocal`,
  `NotLocalReason`, `Resolve`, `SystemResolver`).
- New `ConfigIssue::NotLocal`, which products treat as fatal. `validate` may
  now do DNS lookups; `validate_with` takes the resolver.
- Behaviour change: discovery of an endpoint that fails the guard now
  returns `Unreachable(<NotLocal text>)` where it used to return
  `NotProbed` (`locality = "cloud"` is still `NotProbed`, checked before the
  guard so no lookup is made). The residency broker is guarded too, and
  residency guard lookups time out after 2 s per endpoint (skipped).
- Discovery and residency check `require_local` before contacting an
  endpoint (the `discovery` feature now enables tokio's `rt` for
  `spawn_blocking`).
- `url_locality` is `Cloud` for any non-http(s) scheme with a hierarchical
  part (`unix:/x`, `foo:/evil.com`); `host:port` input is unchanged.
