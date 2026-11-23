# Changelog

## [0.3.0] - Unreleased

Breaking.

### Changed

- The `Cloud` tier is now `Locality::Shared`: since 0.2.0 every model URL
  must be local, so the tier only labels a local endpoint the operator
  trusts less. TOML and JSON write `"shared"`; `"cloud"` is still read, as
  an alias, until 3.0. **Serialized `RoutingConfig` output changes from
  `"cloud"` to `"shared"`: a 2.0 product cannot read a config that a 2.1
  product wrote with `"shared"`.** Stable-surface snapshots in the products
  need regenerating.
- `Policy::allow_cloud` is `Policy::allow_shared`; `Router::with_allow_cloud`
  is `Router::with_allow_shared`.
- `url_is_local(url) -> bool` replaces `url_locality` (same rules, no DNS).
- Pin-fallback reason reads "it's on a shared server"; `ConfigIssue` text and
  docs say "shared". `NotLocalReason::CloudHost` and its `cloud host` text
  are unchanged.
- `LocalOnlyDns` is no longer a unit struct (it holds a resolver): build it
  with `LocalOnlyDns::new()` or `LocalOnlyDns::with_resolver(Arc<dyn Resolve>)`. It is also no longer
  `Copy`.
- `NotLocalReason::NotUnicast` is a new variant: exhaustive matches on
  `NotLocalReason` stop compiling (see Added).

### Added

- `NotLocalReason::NotUnicast(IpAddr)`, "multicast or broadcast address
  <ip>": IPv4 `224.0.0.0/4`, `255.255.255.255` and IPv6 `ff00::/8` are
  refused with it instead of `PublicAddress`.

### Fixed

- `NotLocal.url` from `resolve_local_with` / `LocalOnlyDns` showed `host:0`
  and unbracketed IPv6; it is now `host`, `host:port` or `[v6]:port`.

### Removed

- `url_locality` (use `url_is_local`), `EndpointKind::locality()` and
  `Policy::allow_cloud` / `Router::with_allow_cloud` (see above).

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
  .redirect(Policy::none()).no_proxy()`; connections to a name can then only go to local
  addresses whatever DNS does after the check. reqwest does not call the
  resolver for IP-literal URLs, so keep using `require_local` too.
- An `in_process` default counts as in-process only while its `base_url` is
  unset: setting the public `base_url` field directly makes it an ordinary
  endpoint (`Cloud` unless local, checked by `validate`).
- A bare path such as `/v1/chat` is no longer read as the host `v1`; it is
  `NotHttp`.
- New `DefaultEndpoint::in_process(name)` (+ `is_in_process()`): the
  product's in-process backend (mistralrs). No URL, `Local` by construction;
  `validate`, `verify_local` and discovery never look it up or contact it
  (discovery only walks `[routing.endpoints]`, never the default), and
  routing selects its models as local. Not in TOML. No existing rule is
  widened: an ordinary endpoint with no URL stays `Cloud`, even with
  `locality = "local"`. Replaces placeholder-URL workarounds.
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
