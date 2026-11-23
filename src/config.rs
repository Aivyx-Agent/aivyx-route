//! The `[routing]` config section. Both products embed this shape.

use std::collections::{BTreeMap, BTreeSet};

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::guard::{Resolve, SystemResolver, require_local_with};
use crate::locality::{url_host, url_is_local};
use crate::profile::{CapabilitySet, EndpointRef, Locality, Strength, Tier};
use crate::requirements::TaskOverrides;

/// The `[routing]` section. Unknown keys are ignored so each product can
/// add its own sub-tables (e.g. `aivyx-pa`'s `[routing.escalation]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RoutingConfig {
    /// `false` (the default) means today's single-model behavior.
    pub enabled: bool,
    pub discover: bool,
    pub endpoints: BTreeMap<String, EndpointConfig>,
    pub models: Vec<RosterEntry>,
    pub tasks: TaskOverrides,
    /// Host GPU memory in bytes, for residency scoring when no
    /// `aivyx-broker` reports it. `None`: no won't-fit term without a broker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_bytes: Option<u64>,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        RoutingConfig {
            enabled: false,
            discover: true,
            endpoints: BTreeMap::new(),
            models: Vec::new(),
            tasks: TaskOverrides::default(),
            vram_bytes: None,
        }
    }
}

/// The product's own default backend: where roster entries without an
/// `endpoint` live. Its locality follows the same rule as a
/// `[routing.endpoints]` entry's ([`EndpointConfig::effective_locality`]),
/// so the product passes the backend's URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultEndpoint {
    pub name: EndpointRef,
    pub kind: EndpointKind,
    /// The backend's URL; `None` falls back to the kind's default.
    pub base_url: Option<String>,
    /// Operator override of the locality the address implies, as on an
    /// endpoint.
    pub locality: Option<Locality>,
    /// The URL [`RoutingConfig::verify_local`] confirmed with
    /// `require_local`; never from config. Excluded from `==`.
    pub(crate) dns_verified: DnsVerified,
    /// An in-process backend: no URL, local by construction. Only
    /// [`DefaultEndpoint::in_process`] sets it.
    pub(crate) in_process: bool,
}

/// The exact URL that passed `require_local` at load (see
/// [`RoutingConfig::verify_local`]), or none. A `locality = "local"`
/// override is honoured only while it equals the current `base_url`, so
/// changing the URL after verification fails closed.
///
/// Comparison is deliberately ignored: `PartialEq` is always `true`, so
/// verification state (not config) never makes two configs unequal, and the
/// config round-trip tests and products' "did the config change" checks
/// compare configuration only.
#[derive(Debug, Clone, Default)]
pub(crate) struct DnsVerified(Option<String>);

impl PartialEq for DnsVerified {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for DnsVerified {}

impl DnsVerified {
    fn matches(&self, base_url: Option<&str>) -> bool {
        base_url.is_some() && self.0.as_deref() == base_url
    }
}

impl DefaultEndpoint {
    /// A default endpoint with no URL of its own and no override.
    pub fn new(name: impl Into<String>, kind: EndpointKind) -> Self {
        DefaultEndpoint {
            name: EndpointRef::new(name),
            kind,
            base_url: None,
            locality: None,
            dns_verified: DnsVerified::default(),
            in_process: false,
        }
    }

    /// The product's in-process model backend (e.g. mistralrs): no URL,
    /// nothing to contact, **`Local` by construction**. `validate`,
    /// `verify_local` and discovery never look it up; routing selects its
    /// models as local. This is the only way to get a URL-less local
    /// endpoint: any ordinary endpoint with no URL (even with
    /// `locality = "local"`) is still `Shared`. A `locality = "shared"`
    /// override still applies; [`with_base_url`](Self::with_base_url) makes
    /// it an ordinary endpoint again.
    pub fn in_process(name: impl Into<String>) -> Self {
        DefaultEndpoint {
            in_process: true,
            ..DefaultEndpoint::new(name, EndpointKind::OpenaiCompat)
        }
    }

    /// Whether this is an [`in_process`](Self::in_process) backend.
    pub fn is_in_process(&self) -> bool {
        self.in_process && self.base_url.is_none()
    }

    /// With `base_url` as its URL (which makes an in-process default an
    /// ordinary endpoint).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self.in_process = false;
        self
    }

    /// With a `locality` override.
    #[must_use]
    pub fn with_locality(mut self, locality: Locality) -> Self {
        self.locality = Some(locality);
        self
    }

    /// Whether [`RoutingConfig::verify_local`] confirmed the current URL.
    pub fn dns_verified(&self) -> bool {
        self.dns_verified.matches(self.base_url())
    }

    /// The configured URL, else the kind's default.
    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref().or(self.kind.default_base_url())
    }

    /// Same rule as [`EndpointConfig::effective_locality`].
    pub fn effective_locality(&self) -> Locality {
        if self.is_in_process() && self.locality != Some(Locality::Shared) {
            return Locality::Local;
        }
        effective_locality(self.base_url(), self.locality, self.dns_verified())
    }
}

/// A problem [`RoutingConfig::validate`] found. Every issue is a warning
/// except [`ConfigIssue::NotLocal`], which products treat as fatal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "issue", rename_all = "snake_case")]
pub enum ConfigIssue {
    /// A roster entry names an endpoint that is neither the default nor in
    /// `[routing.endpoints]`. Its model is never selected.
    UnknownEndpoint { model: String, endpoint: String },
    /// The same (endpoint, id) appears in more than one roster entry.
    DuplicateModel { endpoint: String, model: String },
    /// A local endpoint with no `base_url` and no default for its kind.
    MissingBaseUrl { endpoint: String },
    /// An endpoint whose `base_url` name resolves only to local addresses
    /// but isn't recognisably local by its spelling, with no `locality`
    /// override: its models count as shared. Set `locality = "local"` on the
    /// endpoint to use them.
    NonLocalAddress { endpoint: String, host: String },
    /// A `locality = "local"` override that has no effect: on the endpoint
    /// itself (`model` is `None`) because its address isn't local by
    /// [`url_is_local`] and [`RoutingConfig::verify_local`] didn't pass (or
    /// wasn't called); or on a roster entry for a model on a shared-class
    /// endpoint (a roster override can only lower trust).
    LocalOverrideIgnored {
        endpoint: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },
    /// An endpoint (or the product's default backend) whose `base_url`
    /// fails [`require_local`](crate::require_local): a cloud host, a
    /// public address, a name that doesn't resolve, or a non-`http(s)` URL.
    /// `reason` is the `NotLocal` error text.
    ///
    /// **Products treat this issue as fatal**: aivyx is local-only and
    /// must not talk to such an endpoint.
    NotLocal { endpoint: String, reason: String },
}

impl fmt::Display for ConfigIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigIssue::UnknownEndpoint { model, endpoint } => write!(
                f,
                "model `{model}` names endpoint `{endpoint}`, which is not configured; it will never be selected"
            ),
            ConfigIssue::DuplicateModel { endpoint, model } => {
                write!(
                    f,
                    "model `{model}` on endpoint `{endpoint}` is listed more than once"
                )
            }
            ConfigIssue::MissingBaseUrl { endpoint } => write!(
                f,
                "endpoint `{endpoint}` has no base_url and its kind has no default"
            ),
            ConfigIssue::NonLocalAddress { endpoint, host } => write!(
                f,
                "endpoint `{endpoint}` points at `{host}`, which resolves to local addresses but \
                 isn't recognised as a local name, so its models count as shared; set \
                 `locality = \"local\"` on the endpoint to use them"
            ),
            ConfigIssue::NotLocal { endpoint, reason } => {
                write!(f, "endpoint `{endpoint}`: {reason}")
            }
            ConfigIssue::LocalOverrideIgnored {
                endpoint,
                model: None,
            } => write!(
                f,
                "endpoint `{endpoint}`: `locality = \"local\"` has no effect: its address isn't \
                 local and wasn't verified. A name counts as local when `require_local`'s DNS \
                 rule passes (it resolves only to local addresses) and `verify_local` ran at \
                 load"
            ),
            ConfigIssue::LocalOverrideIgnored {
                endpoint,
                model: Some(model),
            } => write!(
                f,
                "model `{model}` is on shared endpoint `{endpoint}`; its `locality = \"local\"` is \
                 ignored"
            ),
        }
    }
}

impl RoutingConfig {
    /// The locality of the endpoint named `name`: the default endpoint's
    /// [`effective_locality`](DefaultEndpoint::effective_locality), or a `[routing.endpoints]` entry's
    /// [`effective_locality`](EndpointConfig::effective_locality). `None`
    /// for a name that is neither (callers fail closed to `Shared`).
    pub fn endpoint_locality(&self, default: &DefaultEndpoint, name: &str) -> Option<Locality> {
        if name == default.name.as_str() {
            Some(default.effective_locality())
        } else {
            self.endpoints
                .get(name)
                .map(EndpointConfig::effective_locality)
        }
    }

    /// Report configuration mistakes; the default endpoint's issues come
    /// first, then other endpoints' (in name order), then roster issues (in
    /// roster order).
    ///
    /// **May do blocking DNS lookups**: each endpoint URL is checked with
    /// [`require_local`](crate::require_local) (system resolver), which
    /// resolves a host name that isn't recognisably local. Inside an async
    /// runtime call it from `spawn_blocking`, or use
    /// [`validate_with`](Self::validate_with) with your own resolver.
    pub fn validate(&self, default: &DefaultEndpoint) -> Vec<ConfigIssue> {
        self.validate_with(default, &SystemResolver)
    }

    /// Load-time pass: for each endpoint (and the default) with
    /// `locality = "local"` whose address isn't local by spelling, run
    /// `require_local` (system resolver); on success record that exact URL
    /// as verified so the override is honoured, on failure report
    /// [`ConfigIssue::NotLocal`] (fatal). **Products must call this at
    /// load**, before using `effective_locality`; an unverified config, or
    /// one whose `base_url` changed since, fails closed. Blocking DNS: inside
    /// an async runtime use `spawn_blocking`, or
    /// [`verify_local_with`](Self::verify_local_with).
    pub fn verify_local(&mut self, default: &mut DefaultEndpoint) -> Vec<ConfigIssue> {
        self.verify_local_with(default, &SystemResolver)
    }

    /// [`verify_local`](Self::verify_local) with an explicit resolver.
    pub fn verify_local_with(
        &mut self,
        default: &mut DefaultEndpoint,
        r: &dyn Resolve,
    ) -> Vec<ConfigIssue> {
        let mut issues = Vec::new();
        let mut one = |name: &str, url: Option<&str>, locality, flag: &mut DnsVerified| {
            flag.0 = None;
            let Some(url) = url.filter(|u| locality == Some(Locality::Local) && !url_is_local(u))
            else {
                return;
            };
            match require_local_with(url, r) {
                Ok(()) => flag.0 = Some(url.to_owned()),
                Err(e) => issues.push(ConfigIssue::NotLocal {
                    endpoint: name.to_owned(),
                    reason: e.to_string(),
                }),
            }
        };
        let url = default.base_url().map(str::to_owned);
        one(
            default.name.as_str(),
            url.as_deref(),
            default.locality,
            &mut default.dns_verified,
        );
        for (name, c) in &mut self.endpoints {
            let url = c.base_url().map(str::to_owned);
            one(name, url.as_deref(), c.locality, &mut c.dns_verified);
        }
        issues
    }

    /// [`validate`](Self::validate) with an explicit resolver.
    pub fn validate_with(&self, default: &DefaultEndpoint, r: &dyn Resolve) -> Vec<ConfigIssue> {
        let mut issues = Vec::new();
        // An in-process backend has no URL to check.
        if !default.is_in_process() {
            endpoint_issues(
                default.name.as_str(),
                default.base_url(),
                default.locality,
                default.dns_verified(),
                r,
                &mut issues,
            );
        }
        for (name, c) in &self.endpoints {
            endpoint_issues(
                name,
                c.base_url(),
                c.locality,
                c.dns_verified(),
                r,
                &mut issues,
            );
        }
        let mut seen = BTreeSet::new();
        for entry in &self.models {
            let endpoint = entry
                .endpoint
                .clone()
                .unwrap_or_else(|| default.name.as_str().to_owned());
            match self.endpoint_locality(default, &endpoint) {
                None => issues.push(ConfigIssue::UnknownEndpoint {
                    model: entry.id.clone(),
                    endpoint: endpoint.clone(),
                }),
                Some(Locality::Shared) if entry.locality == Some(Locality::Local) => {
                    issues.push(ConfigIssue::LocalOverrideIgnored {
                        endpoint: endpoint.clone(),
                        model: Some(entry.id.clone()),
                    });
                }
                Some(_) => {}
            }
            if !seen.insert((endpoint.clone(), entry.id.clone())) {
                issues.push(ConfigIssue::DuplicateModel {
                    endpoint,
                    model: entry.id.clone(),
                });
            }
        }
        issues
    }
}

/// Locality issues and a missing URL for one endpoint.
fn endpoint_issues(
    name: &str,
    base_url: Option<&str>,
    locality: Option<Locality>,
    dns_verified: bool,
    r: &dyn Resolve,
    issues: &mut Vec<ConfigIssue>,
) {
    let Some(url) = base_url else {
        issues.push(ConfigIssue::MissingBaseUrl {
            endpoint: name.to_owned(),
        });
        return;
    };
    let by_spelling = url_is_local(url);
    match require_local_with(url, r) {
        Err(e) => issues.push(ConfigIssue::NotLocal {
            endpoint: name.to_owned(),
            reason: e.to_string(),
        }),
        Ok(()) if !by_spelling && locality.is_none() => {
            issues.push(ConfigIssue::NonLocalAddress {
                endpoint: name.to_owned(),
                host: url_host(url).unwrap_or_else(|| url.to_owned()),
            });
        }
        Ok(()) => {}
    }
    if locality == Some(Locality::Local) && !by_spelling && !dns_verified {
        issues.push(ConfigIssue::LocalOverrideIgnored {
            endpoint: name.to_owned(),
            model: None,
        });
    }
}

/// The one locality rule for an endpoint, default or configured, in order:
/// `locality = "shared"` is `Shared`; a local address ([`url_is_local`]) is
/// `Local`; `locality = "local"` with `dns_verified` (verified by
/// [`RoutingConfig::verify_local`]) is `Local`; anything else, including no
/// URL or an unverified override, is `Shared` (fail closed).
fn effective_locality(
    base_url: Option<&str>,
    locality: Option<Locality>,
    dns_verified: bool,
) -> Locality {
    if locality == Some(Locality::Shared) {
        return Locality::Shared;
    }
    if base_url.is_some_and(url_is_local) || (locality == Some(Locality::Local) && dns_verified) {
        return Locality::Local;
    }
    Locality::Shared
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointKind {
    Ollama,
    /// `llama-server` in router mode (multi-model, `GET /models`).
    LlamaRouter,
    /// Anything serving `GET /v1/models` (single-model llama-server, Jan,
    /// vLLM, …).
    OpenaiCompat,
    /// Lemonade Server: single-LLM-at-a-time, with its own residency API.
    Lemonade,
}

impl EndpointKind {
    pub fn default_base_url(self) -> Option<&'static str> {
        match self {
            EndpointKind::Ollama => Some("http://localhost:11434"),
            EndpointKind::LlamaRouter => Some("http://localhost:8080"),
            EndpointKind::Lemonade => Some("http://127.0.0.1:13305/api"),
            EndpointKind::OpenaiCompat => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointConfig {
    pub kind: EndpointKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Operator override of the locality the address implies: `shared` marks
    /// the endpoint shared. `local` is honoured only for a LAN box with a DNS
    /// name, and only after [`RoutingConfig::verify_local`] confirmed that
    /// [`require_local`](crate::require_local) passes (it resolves only to
    /// local addresses); otherwise it has no effect and
    /// [`RoutingConfig::validate`] reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locality: Option<Locality>,
    /// The URL [`RoutingConfig::verify_local`] confirmed with
    /// `require_local`; never from TOML. Excluded from `==`
    /// ([`DnsVerified`]).
    #[serde(skip)]
    pub(crate) dns_verified: DnsVerified,
}

impl EndpointConfig {
    /// An endpoint of `kind` at `base_url` (`None`: the kind's default),
    /// with no override.
    pub fn new(kind: EndpointKind, base_url: Option<String>) -> Self {
        EndpointConfig {
            kind,
            base_url,
            locality: None,
            dns_verified: DnsVerified::default(),
        }
    }

    /// With a `locality` override.
    #[must_use]
    pub fn with_locality(mut self, locality: Locality) -> Self {
        self.locality = Some(locality);
        self
    }

    /// Whether [`RoutingConfig::verify_local`] confirmed the current URL
    /// (so a `locality = "local"` override is honoured). Changing
    /// `base_url` afterwards makes this `false` again.
    pub fn dns_verified(&self) -> bool {
        self.dns_verified.matches(self.base_url())
    }

    /// The configured URL, else the kind's default.
    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref().or(self.kind.default_base_url())
    }

    /// Whether requests to this endpoint leave the machine's network.
    /// `locality = "shared"` is `Shared`; otherwise the
    /// [`base_url`](Self::base_url) host decides ([`url_is_local`]); no URL
    /// at all is `Shared` (fail closed). See [`effective_locality`] for the
    /// order; a `local` override counts only once verified.
    pub fn effective_locality(&self) -> Locality {
        effective_locality(self.base_url(), self.locality, self.dns_verified())
    }
}

/// One `[[routing.models]]` entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RosterEntry {
    pub id: String,
    /// Key of `[routing.endpoints]`; `None` = the product's default backend.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// `shared` marks this model shared even on a local endpoint. `local`
    /// never makes a model on a shared endpoint local (it is ignored, and
    /// [`RoutingConfig::validate`] reports it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locality: Option<Locality>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<Tier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strengths: Option<BTreeSet<Strength>>,
    /// Operator tie-break; higher wins. Unset = 0 (or an earlier entry's).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,
    /// Added to what discovery found.
    #[serde(default)]
    pub capabilities: CapabilitySet,
    /// Removed from what discovery found (e.g. unreliable tool calling).
    #[serde(default)]
    pub capabilities_deny: CapabilitySet,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::Capability;

    #[derive(Deserialize)]
    struct Doc {
        routing: RoutingConfig,
    }

    const SPEC_EXAMPLE: &str = r#"
[routing]
enabled = true
discover = true

[routing.endpoints.ollama-main]
kind = "ollama"
base_url = "http://localhost:11434"

[[routing.models]]
id = "qwen3-coder:30b"
endpoint = "ollama-main"
tier = "large"
strengths = ["code", "reasoning"]
priority = 10

[[routing.models]]
id = "llava:13b"
tier = "small"
capabilities = ["vision"]
capabilities_deny = ["tools"]

[routing.tasks]
summarize = { tier = "small" }
"#;

    #[test]
    fn parses_the_spec_example() {
        let c = toml::from_str::<Doc>(SPEC_EXAMPLE).unwrap().routing;
        assert!(c.enabled);
        assert!(c.discover);
        let ep = &c.endpoints["ollama-main"];
        assert_eq!(ep.kind, EndpointKind::Ollama);
        assert_eq!(ep.base_url(), Some("http://localhost:11434"));
        assert_eq!(c.models.len(), 2);
        let qwen = &c.models[0];
        assert_eq!(qwen.id, "qwen3-coder:30b");
        assert_eq!(qwen.endpoint.as_deref(), Some("ollama-main"));
        assert_eq!(qwen.tier, Some(Tier::Large));
        assert_eq!(
            qwen.strengths,
            Some(BTreeSet::from([Strength::Code, Strength::Reasoning]))
        );
        assert_eq!(qwen.priority, Some(10));
        let llava = &c.models[1];
        assert_eq!(llava.endpoint, None);
        assert_eq!(
            llava.capabilities,
            CapabilitySet::from([Capability::Vision])
        );
        assert_eq!(
            llava.capabilities_deny,
            CapabilitySet::from([Capability::Tools])
        );
        assert_eq!(c.tasks.0["summarize"].tier, Some(Tier::Small));
    }

    #[test]
    fn defaults_are_off_with_discovery_on() {
        let c = RoutingConfig::default();
        assert!(!c.enabled);
        assert!(c.discover);
        assert!(c.endpoints.is_empty() && c.models.is_empty() && c.tasks.0.is_empty());
        let c = toml::from_str::<Doc>("[routing]\n").unwrap().routing;
        assert_eq!(c, RoutingConfig::default());
    }

    #[test]
    fn product_specific_keys_are_ignored() {
        let c = toml::from_str::<Doc>(
            "[routing]\nenabled = true\n[routing.escalation]\nmode = \"ask\"\n",
        )
        .unwrap()
        .routing;
        assert!(c.enabled);
    }

    #[test]
    fn typos_in_model_entries_are_rejected() {
        let err = toml::from_str::<Doc>("[[routing.models]]\nid = \"x\"\nteir = \"small\"\n");
        assert!(err.is_err());
    }

    #[test]
    fn endpoint_kinds_know_default_urls() {
        assert_eq!(
            EndpointKind::Ollama.default_base_url(),
            Some("http://localhost:11434")
        );
        assert_eq!(
            EndpointKind::LlamaRouter.default_base_url(),
            Some("http://localhost:8080")
        );
        assert_eq!(EndpointKind::OpenaiCompat.default_base_url(), None);
        let ep: EndpointConfig = toml::from_str("kind = \"llama_router\"").unwrap();
        assert_eq!(ep.base_url(), Some("http://localhost:8080"));
        let ep: EndpointConfig = toml::from_str("kind = \"openai_compat\"").unwrap();
        assert_eq!(ep.base_url(), None);
    }

    #[test]
    fn lemonade_is_local_with_its_own_default_port() {
        assert_eq!(
            EndpointKind::Lemonade.default_base_url(),
            Some("http://127.0.0.1:13305/api")
        );
        let ep: EndpointConfig = toml::from_str("kind = \"lemonade\"").unwrap();
        assert_eq!(ep.kind, EndpointKind::Lemonade);
        assert_eq!(ep.base_url(), Some("http://127.0.0.1:13305/api"));
    }

    #[test]
    fn roster_entry_locality_is_explicit_and_optional() {
        let e: RosterEntry = toml::from_str("id = \"claude\"\nlocality = \"cloud\"").unwrap();
        assert_eq!(e.locality, Some(Locality::Shared));
        let e: RosterEntry = toml::from_str("id = \"x\"").unwrap();
        assert_eq!(e.locality, None);
        assert_eq!(e.priority, None);
        assert_eq!(e.strengths, None);
    }

    fn local_default() -> DefaultEndpoint {
        DefaultEndpoint {
            name: EndpointRef::new("main"),
            kind: EndpointKind::Ollama,
            base_url: None,
            locality: None,
            dns_verified: Default::default(),
            in_process: false,
        }
    }

    #[test]
    fn a_clean_config_has_no_issues() {
        let c = toml::from_str::<Doc>(SPEC_EXAMPLE).unwrap().routing;
        assert_eq!(c.validate(&local_default()), vec![]);
    }

    #[test]
    fn validate_reports_every_issue_kind_in_order() {
        let c = toml::from_str::<Doc>(
            r#"
[routing.endpoints.jan]
kind = "openai_compat"

[routing.endpoints.gpu]
kind = "ollama"

[[routing.models]]
id = "a"
endpoint = "olama"

[[routing.models]]
id = "b"

[[routing.models]]
id = "b"
endpoint = "main"

[[routing.models]]
id = "b"
endpoint = "gpu"
"#,
        )
        .unwrap()
        .routing;
        let issues = c.validate(&local_default());
        assert_eq!(
            issues,
            vec![
                ConfigIssue::MissingBaseUrl {
                    endpoint: "jan".into()
                },
                ConfigIssue::UnknownEndpoint {
                    model: "a".into(),
                    endpoint: "olama".into()
                },
                ConfigIssue::DuplicateModel {
                    endpoint: "main".into(),
                    model: "b".into()
                },
            ]
        );
        let text: Vec<String> = issues.iter().map(ToString::to_string).collect();
        assert_eq!(
            text,
            vec![
                "endpoint `jan` has no base_url and its kind has no default",
                "model `a` names endpoint `olama`, which is not configured; it will never be selected",
                "model `b` on endpoint `main` is listed more than once",
            ]
        );
    }

    fn endpoint(toml_src: &str) -> EndpointConfig {
        toml::from_str(toml_src).unwrap()
    }

    #[test]
    fn a_hosted_openai_compatible_endpoint_is_shared() {
        let groq =
            endpoint("kind = \"openai_compat\"\nbase_url = \"https://api.groq.com/openai/v1\"");
        assert_eq!(groq.effective_locality(), Locality::Shared);
    }

    #[test]
    fn local_addresses_keep_non_shared_kinds_local() {
        for url in [
            "http://127.0.0.1:8080",
            "http://10.0.0.5:11434",
            "http://192.168.1.20:8080/v1",
            "http://100.100.1.2:11434",
            "http://[::1]:8080",
            "http://gpu.lan:8080",
            "http://myhost:8080",
        ] {
            for kind in ["ollama", "llama_router", "openai_compat", "lemonade"] {
                let ep = endpoint(&format!("kind = \"{kind}\"\nbase_url = \"{url}\""));
                assert_eq!(ep.effective_locality(), Locality::Local, "{kind} {url}");
            }
        }
        // Kind defaults are loopback.
        assert_eq!(
            endpoint("kind = \"ollama\"").effective_locality(),
            Locality::Local
        );
        assert_eq!(
            endpoint("kind = \"lemonade\"").effective_locality(),
            Locality::Local
        );
    }

    #[test]
    fn an_endpoint_locality_override_can_only_lower_trust() {
        let public = endpoint(
            "kind = \"ollama\"\nbase_url = \"http://gpu.example.com:11434\"\nlocality = \"local\"",
        );
        assert_eq!(public.effective_locality(), Locality::Shared);
        let forced = endpoint(
            "kind = \"ollama\"\nbase_url = \"http://localhost:11434\"\nlocality = \"cloud\"",
        );
        assert_eq!(forced.effective_locality(), Locality::Shared);
        // `local` on a local address is a no-op.
        let same =
            endpoint("kind = \"ollama\"\nbase_url = \"http://127.0.0.1:9\"\nlocality = \"local\"");
        assert_eq!(same.effective_locality(), Locality::Local);
    }

    #[test]
    fn removed_cloud_kinds_fail_to_parse() {
        for kind in ["anthropic", "openai"] {
            let err = toml::from_str::<EndpointConfig>(&format!("kind = \"{kind}\"")).unwrap_err();
            assert!(err.to_string().contains("unknown variant"), "{kind}: {err}");
        }
    }

    #[test]
    fn a_non_shared_kind_with_no_address_fails_closed() {
        assert_eq!(
            endpoint("kind = \"openai_compat\"").effective_locality(),
            Locality::Shared
        );
    }

    /// Resolves only what it is told; anything else fails (no network).
    struct Stub(Vec<(&'static str, &'static str)>);

    impl Resolve for Stub {
        fn resolve(&self, host: &str, _port: u16) -> std::io::Result<Vec<std::net::IpAddr>> {
            self.0
                .iter()
                .find(|(h, _)| *h == host)
                .map(|(_, ip)| vec![ip.parse().unwrap()])
                .ok_or_else(|| std::io::Error::other("no such host"))
        }
    }

    fn stub() -> Stub {
        Stub(vec![
            ("lan-box.example.com", "192.168.1.20"),
            ("pub.example.com", "203.0.113.7"),
        ])
    }

    #[test]
    fn validate_explains_locality_surprises() {
        let c = toml::from_str::<Doc>(
            r#"
[routing.endpoints.groq]
kind = "openai_compat"
base_url = "https://api.groq.com/openai/v1"

[routing.endpoints.lan]
kind = "ollama"
base_url = "http://pub.example.com:11434"
locality = "local"

[routing.endpoints.dns-lan]
kind = "ollama"
base_url = "http://lan-box.example.com:11434"

[routing.endpoints.home]
kind = "ollama"
base_url = "http://192.168.1.20:11434"

[routing.endpoints.ftp]
kind = "ollama"
base_url = "ftp://127.0.0.1"

[routing.endpoints.gone]
kind = "ollama"
base_url = "http://nonexistent.example.com"
"#,
        )
        .unwrap()
        .routing;
        let issues = c.validate_with(&local_default(), &stub());
        let not_local = |endpoint: &str, why: &str| ConfigIssue::NotLocal {
            endpoint: endpoint.into(),
            reason: why.into(),
        };
        assert_eq!(
            issues,
            vec![
                // BTreeMap order: dns-lan, ftp, gone, groq, home, lan.
                ConfigIssue::NonLocalAddress {
                    endpoint: "dns-lan".into(),
                    host: "lan-box.example.com".into()
                },
                not_local(
                    "ftp",
                    "ftp://127.0.0.1/ is not local (not an http(s) URL) — aivyx is local-only; point it at a model server on this machine or your own network"
                ),
                not_local(
                    "gone",
                    "http://nonexistent.example.com/ is not local (doesn't resolve) — aivyx is local-only; point it at a model server on this machine or your own network"
                ),
                not_local(
                    "groq",
                    "https://api.groq.com/openai/v1 is not local (cloud host) — aivyx is local-only; point it at a model server on this machine or your own network"
                ),
                not_local(
                    "lan",
                    "http://pub.example.com:11434/ is not local (resolves to a public address 203.0.113.7) — aivyx is local-only; point it at a model server on this machine or your own network"
                ),
                ConfigIssue::LocalOverrideIgnored {
                    endpoint: "lan".into(),
                    model: None
                },
            ]
        );
        let text: Vec<String> = issues.iter().map(ToString::to_string).collect();
        assert!(
            text[0].contains("set `locality = \"local\"`"),
            "{}",
            text[0]
        );
        assert!(
            text[5].contains("no effect") && text[5].contains("`require_local`'s DNS rule"),
            "{}",
            text[5]
        );
    }

    #[test]
    fn verify_local_honours_a_local_override_only_after_the_guard_passes() {
        let src = |host: &str, locality: &str| {
            toml::from_str::<Doc>(&format!(
                "[routing.endpoints.box]\nkind = \"ollama\"\nbase_url = \"http://{host}:11434\"\n{locality}"
            ))
            .unwrap()
            .routing
        };
        let over = "locality = \"local\"";
        // Override + DNS name resolving to the LAN: Local after verify.
        let mut c = src("lan-box.example.com", over);
        let mut d = local_default();
        assert_eq!(c.endpoints["box"].effective_locality(), Locality::Shared); // unverified
        assert_eq!(
            c.validate_with(&d, &stub()),
            vec![ConfigIssue::LocalOverrideIgnored {
                endpoint: "box".into(),
                model: None
            }]
        );
        assert_eq!(c.verify_local_with(&mut d, &stub()), vec![]);
        assert!(c.endpoints["box"].dns_verified());
        assert_eq!(c.endpoints["box"].effective_locality(), Locality::Local);
        assert_eq!(c.endpoint_locality(&d, "box"), Some(Locality::Local));
        assert_eq!(c.validate_with(&d, &stub()), vec![]);
        // No override: Shared, with the hint.
        let mut c = src("lan-box.example.com", "");
        assert_eq!(c.verify_local_with(&mut d, &stub()), vec![]);
        assert_eq!(c.endpoints["box"].effective_locality(), Locality::Shared);
        assert!(matches!(
            c.validate_with(&d, &stub()).as_slice(),
            [ConfigIssue::NonLocalAddress { .. }]
        ));
        // Override, but the name resolves public: NotLocal, stays Shared.
        let mut c = src("pub.example.com", over);
        let issues = c.verify_local_with(&mut d, &stub());
        assert!(
            matches!(issues.as_slice(), [ConfigIssue::NotLocal { endpoint, .. }] if endpoint == "box")
        );
        assert_eq!(c.endpoints["box"].effective_locality(), Locality::Shared);
        // A shared override wins over everything.
        let mut c = src("127.0.0.1", "locality = \"cloud\"");
        assert_eq!(c.verify_local_with(&mut d, &stub()), vec![]);
        assert_eq!(c.endpoints["box"].effective_locality(), Locality::Shared);
        // The default endpoint is verified too, and dns_verified never reaches TOML.
        let mut c = RoutingConfig::default();
        let mut d = default_at(
            EndpointKind::Ollama,
            Some("http://lan-box.example.com:11434"),
            Some(Locality::Local),
        );
        assert_eq!(c.verify_local_with(&mut d, &stub()), vec![]);
        assert_eq!(d.effective_locality(), Locality::Local);
        let c = src("lan-box.example.com", over);
        assert!(
            !toml::to_string(&c.endpoints["box"])
                .unwrap()
                .contains("dns_verified")
        );
    }

    /// Resolver that fails the test if it is ever asked anything.
    struct Forbidden;

    impl Resolve for Forbidden {
        fn resolve(&self, host: &str, _: u16) -> std::io::Result<Vec<std::net::IpAddr>> {
            panic!("looked up {host}")
        }
    }

    #[test]
    fn in_process_with_a_public_base_url_set_directly_is_shared_and_not_local() {
        let mut d = DefaultEndpoint::in_process("mistralrs");
        d.base_url = Some("https://8.8.8.8/v1".into());
        assert!(!d.is_in_process());
        assert_eq!(d.effective_locality(), Locality::Shared);
        let c = RoutingConfig::default();
        let issues = c.validate_with(&d, &Forbidden);
        assert!(
            issues
                .iter()
                .any(|i| matches!(i, ConfigIssue::NotLocal { .. })),
            "{issues:?}"
        );
    }

    #[test]
    fn an_in_process_default_is_local_without_a_url_or_lookup() {
        let d = DefaultEndpoint::in_process("mistralrs");
        assert!(d.is_in_process());
        assert_eq!(d.base_url(), None);
        assert_eq!(d.effective_locality(), Locality::Local);
        let mut c = RoutingConfig::default();
        assert_eq!(c.endpoint_locality(&d, "mistralrs"), Some(Locality::Local));
        // No MissingBaseUrl, no NotLocal, no lookup.
        assert_eq!(c.validate_with(&d, &Forbidden), vec![]);
        let mut d2 = d.clone();
        assert_eq!(c.verify_local_with(&mut d2, &Forbidden), vec![]);
        assert_eq!(d2.effective_locality(), Locality::Local);
        // Trust can still be lowered; a URL makes it ordinary again.
        assert_eq!(
            d.clone()
                .with_locality(Locality::Shared)
                .effective_locality(),
            Locality::Shared
        );
        assert!(!d.with_base_url("https://api.groq.com").is_in_process());
    }

    #[test]
    fn an_ordinary_endpoint_with_no_url_stays_shared_even_with_a_local_override() {
        let d =
            DefaultEndpoint::new("main", EndpointKind::OpenaiCompat).with_locality(Locality::Local);
        assert_eq!(d.effective_locality(), Locality::Shared);
        let ep =
            EndpointConfig::new(EndpointKind::OpenaiCompat, None).with_locality(Locality::Local);
        assert_eq!(ep.effective_locality(), Locality::Shared);
        let mut c = RoutingConfig::default();
        let mut d = d;
        assert_eq!(c.verify_local_with(&mut d, &stub()), vec![]);
        assert_eq!(d.effective_locality(), Locality::Shared);
    }

    #[test]
    fn verification_goes_stale_when_the_url_changes_and_is_not_compared() {
        let mut c = RoutingConfig::default();
        c.endpoints.insert(
            "box".into(),
            EndpointConfig::new(
                EndpointKind::Ollama,
                Some("http://lan-box.example.com:11434".into()),
            )
            .with_locality(Locality::Local),
        );
        let unverified = c.clone();
        let mut d = DefaultEndpoint::new("main", EndpointKind::Ollama);
        assert_eq!(c.verify_local_with(&mut d, &stub()), vec![]);
        assert!(c.endpoints["box"].dns_verified());
        assert_eq!(c, unverified, "verification state is not part of ==");
        // Pointing it elsewhere afterwards fails closed.
        c.endpoints.get_mut("box").unwrap().base_url = Some("http://pub.example.com:11434".into());
        assert!(!c.endpoints["box"].dns_verified());
        assert_eq!(c.endpoints["box"].effective_locality(), Locality::Shared);
        // Constructors on the default endpoint.
        let d = DefaultEndpoint::new("m", EndpointKind::OpenaiCompat)
            .with_base_url("http://10.0.0.2:1")
            .with_locality(Locality::Shared);
        assert_eq!(d.effective_locality(), Locality::Shared);
    }

    #[test]
    fn validate_reports_a_roster_local_override_on_a_shared_endpoint() {
        let c = toml::from_str::<Doc>(
            r#"
[routing.endpoints.claude]
kind = "openai_compat"
base_url = "http://127.0.0.1:9"
locality = "cloud"

[routing.endpoints.groq]
kind = "openai_compat"
base_url = "http://127.0.0.1:9"
locality = "cloud"

[[routing.models]]
id = "claude-sonnet"
endpoint = "claude"
locality = "local"

[[routing.models]]
id = "llama"
endpoint = "groq"
locality = "local"

[[routing.models]]
id = "fine"
locality = "local"

[[routing.models]]
id = "typo"
endpoint = "olama"
locality = "local"
"#,
        )
        .unwrap()
        .routing;
        let issues = c.validate_with(&local_default(), &stub());
        assert_eq!(
            issues,
            vec![
                ConfigIssue::LocalOverrideIgnored {
                    endpoint: "claude".into(),
                    model: Some("claude-sonnet".into())
                },
                ConfigIssue::LocalOverrideIgnored {
                    endpoint: "groq".into(),
                    model: Some("llama".into())
                },
                ConfigIssue::UnknownEndpoint {
                    model: "typo".into(),
                    endpoint: "olama".into()
                },
            ]
        );
        assert_eq!(
            issues[0].to_string(),
            "model `claude-sonnet` is on shared endpoint `claude`; its `locality = \"local\"` is ignored"
        );
        let shared_default = DefaultEndpoint {
            name: EndpointRef::new("main"),
            kind: EndpointKind::OpenaiCompat,
            base_url: Some("http://127.0.0.1:9".into()),
            locality: Some(Locality::Shared),
            dns_verified: Default::default(),
            in_process: false,
        };
        let c = toml::from_str::<Doc>("[[routing.models]]\nid = \"x\"\nlocality = \"local\"\n")
            .unwrap()
            .routing;
        assert_eq!(
            c.validate_with(&shared_default, &stub()),
            vec![ConfigIssue::LocalOverrideIgnored {
                endpoint: "main".into(),
                model: Some("x".into())
            }]
        );
    }

    fn default_at(
        kind: EndpointKind,
        url: Option<&str>,
        locality: Option<Locality>,
    ) -> DefaultEndpoint {
        DefaultEndpoint {
            base_url: url.map(Into::into),
            locality,
            ..DefaultEndpoint::new("main", kind)
        }
    }

    #[test]
    fn the_default_endpoint_follows_the_endpoint_locality_rule() {
        use EndpointKind::*;
        use Locality::*;
        for (kind, url, locality, want) in [
            (
                OpenaiCompat,
                Some("https://api.groq.com/openai/v1"),
                None,
                Shared,
            ),
            (Ollama, Some("http://gpu.example.com:11434"), None, Shared),
            (OpenaiCompat, Some("http://127.0.0.1:8080"), None, Local),
            (
                OpenaiCompat,
                Some("http://192.168.1.20:8080/v1"),
                None,
                Local,
            ),
            (LlamaRouter, Some("http://gpu.lan:8080"), None, Local),
            // The kind's default URL.
            (Ollama, None, None, Local),
            // No address at all fails closed.
            (OpenaiCompat, None, None, Shared),
            // An override can only lower trust.
            (
                Ollama,
                Some("http://gpu.example.com:11434"),
                Some(Local),
                Shared,
            ),
            (Ollama, Some("http://localhost:11434"), Some(Shared), Shared),
            (Ollama, Some("http://127.0.0.1:9"), Some(Local), Local),
        ] {
            let d = default_at(kind, url, locality);
            assert_eq!(
                d.effective_locality(),
                want,
                "{kind:?} {url:?} {locality:?}"
            );
            assert_eq!(
                RoutingConfig::default().endpoint_locality(&d, "main"),
                Some(want)
            );
        }
    }

    #[test]
    fn validate_reports_default_endpoint_locality_surprises() {
        let c = RoutingConfig::default();
        let hosted = default_at(
            EndpointKind::OpenaiCompat,
            Some("https://api.groq.com/openai/v1"),
            None,
        );
        assert!(matches!(
            c.validate_with(&hosted, &stub()).as_slice(),
            [ConfigIssue::NotLocal { endpoint, reason }]
                if endpoint == "main" && reason.contains("cloud host")
        ));
        let lan = default_at(
            EndpointKind::OpenaiCompat,
            Some("http://10.0.0.2:8080"),
            None,
        );
        assert_eq!(c.validate_with(&lan, &stub()), vec![]);
        let nowhere = default_at(EndpointKind::OpenaiCompat, None, None);
        assert_eq!(
            c.validate_with(&nowhere, &stub()),
            vec![ConfigIssue::MissingBaseUrl {
                endpoint: "main".into()
            }]
        );
        let ignored = default_at(
            EndpointKind::Ollama,
            Some("http://203.0.113.7:11434"),
            Some(Locality::Local),
        );
        assert!(matches!(
            c.validate_with(&ignored, &stub()).as_slice(),
            [
                ConfigIssue::NotLocal { .. },
                ConfigIssue::LocalOverrideIgnored { model: None, .. }
            ]
        ));
    }

    #[test]
    fn routing_config_round_trips_through_toml() {
        #[derive(Serialize)]
        struct Out<'a> {
            routing: &'a RoutingConfig,
        }
        let c = toml::from_str::<Doc>(SPEC_EXAMPLE).unwrap().routing;
        let text = toml::to_string(&Out { routing: &c }).unwrap();
        let back = toml::from_str::<Doc>(&text).unwrap().routing;
        assert_eq!(back, c, "{text}");
    }

    #[test]
    fn vram_bytes_is_optional() {
        let with: RoutingConfig =
            toml::from_str("enabled = true\nvram_bytes = 25769803776\n").unwrap();
        assert_eq!(with.vram_bytes, Some(25_769_803_776));
        let without: RoutingConfig = toml::from_str("enabled = true\n").unwrap();
        assert_eq!(without.vram_bytes, None);
        assert!(!toml::to_string(&without).unwrap().contains("vram_bytes"));
    }
}
