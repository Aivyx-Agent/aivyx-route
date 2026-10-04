//! The `[routing]` config section. Both products embed this shape.

use std::collections::{BTreeMap, BTreeSet};

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::locality::{url_host, url_locality};
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
/// `endpoint` live. Its kind decides their locality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultEndpoint {
    pub name: EndpointRef,
    pub kind: EndpointKind,
}

/// A problem [`RoutingConfig::validate`] found. None is fatal; products
/// should warn at startup.
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
    /// A non-cloud-kind endpoint whose `base_url` host is not a local
    /// address, with no explicit `locality`: its models count as cloud.
    NonLocalAddress { endpoint: String, host: String },
    /// A `locality = "local"` override on something cloud, which is
    /// ignored: on the endpoint itself (`model` is `None`), or on a roster
    /// entry for a model on a cloud endpoint.
    LocalOverrideIgnored {
        endpoint: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },
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
                "endpoint `{endpoint}` points at `{host}`, which is not a local address, so its \
                 models count as cloud; if it is on your own network, set `locality = \"local\"` \
                 on the endpoint"
            ),
            ConfigIssue::LocalOverrideIgnored {
                endpoint,
                model: None,
            } => write!(
                f,
                "endpoint `{endpoint}` is a cloud endpoint; its `locality = \"local\"` is ignored"
            ),
            ConfigIssue::LocalOverrideIgnored {
                endpoint,
                model: Some(model),
            } => write!(
                f,
                "model `{model}` is on cloud endpoint `{endpoint}`; its `locality = \"local\"` is \
                 ignored"
            ),
        }
    }
}

impl RoutingConfig {
    /// Report configuration mistakes. Pure; endpoint issues come first (in
    /// name order), then roster issues (in roster order).
    pub fn validate(&self, default: &DefaultEndpoint) -> Vec<ConfigIssue> {
        let mut issues = Vec::new();
        for (name, c) in &self.endpoints {
            let cloud_kind = c.kind.locality() == Locality::Cloud;
            match c.base_url() {
                None if !cloud_kind => issues.push(ConfigIssue::MissingBaseUrl {
                    endpoint: name.clone(),
                }),
                Some(url)
                    if !cloud_kind
                        && c.locality.is_none()
                        && url_locality(url) == Locality::Cloud =>
                {
                    issues.push(ConfigIssue::NonLocalAddress {
                        endpoint: name.clone(),
                        host: url_host(url).unwrap_or_default(),
                    });
                }
                _ => {}
            }
            if cloud_kind && c.locality == Some(Locality::Local) {
                issues.push(ConfigIssue::LocalOverrideIgnored {
                    endpoint: name.clone(),
                    model: None,
                });
            }
        }
        let mut seen = BTreeSet::new();
        for entry in &self.models {
            let endpoint = entry
                .endpoint
                .clone()
                .unwrap_or_else(|| default.name.as_str().to_owned());
            if endpoint != default.name.as_str() && !self.endpoints.contains_key(&endpoint) {
                issues.push(ConfigIssue::UnknownEndpoint {
                    model: entry.id.clone(),
                    endpoint: endpoint.clone(),
                });
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
    Anthropic,
    Openai,
}

impl EndpointKind {
    /// The kind's own locality: `Cloud` for hosted APIs. A non-cloud kind
    /// can still point at a hosted server, so decisions about where
    /// requests go use [`EndpointConfig::effective_locality`] instead.
    pub fn locality(self) -> Locality {
        match self {
            EndpointKind::Anthropic | EndpointKind::Openai => Locality::Cloud,
            EndpointKind::Ollama
            | EndpointKind::LlamaRouter
            | EndpointKind::OpenaiCompat
            | EndpointKind::Lemonade => Locality::Local,
        }
    }

    pub fn default_base_url(self) -> Option<&'static str> {
        match self {
            EndpointKind::Ollama => Some("http://localhost:11434"),
            EndpointKind::LlamaRouter => Some("http://localhost:8080"),
            EndpointKind::Lemonade => Some("http://127.0.0.1:13305/api"),
            EndpointKind::OpenaiCompat | EndpointKind::Anthropic | EndpointKind::Openai => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointConfig {
    pub kind: EndpointKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Operator override of the locality the address implies: `local`
    /// marks a non-cloud kind local even when its host looks public (a LAN
    /// box with a public DNS name); `cloud` marks it cloud. Never makes a
    /// cloud kind local.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locality: Option<Locality>,
}

impl EndpointConfig {
    /// The configured URL, else the kind's default.
    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref().or(self.kind.default_base_url())
    }

    /// Whether requests to this endpoint leave the machine's network.
    /// Cloud kinds are always `Cloud`. Otherwise an explicit `locality`
    /// wins, else the [`base_url`](Self::base_url) host decides
    /// ([`url_locality`]); no URL at all is `Cloud` (fail closed).
    pub fn effective_locality(&self) -> Locality {
        if self.kind.locality() == Locality::Cloud {
            return Locality::Cloud;
        }
        match self.locality {
            Some(locality) => locality,
            None => self.base_url().map_or(Locality::Cloud, url_locality),
        }
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
    /// Overrides the locality implied by the endpoint kind.
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
    fn endpoint_kinds_know_locality_and_default_urls() {
        assert_eq!(EndpointKind::Anthropic.locality(), Locality::Cloud);
        assert_eq!(EndpointKind::Openai.locality(), Locality::Cloud);
        assert_eq!(EndpointKind::Ollama.locality(), Locality::Local);
        assert_eq!(EndpointKind::OpenaiCompat.locality(), Locality::Local);
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
        assert_eq!(EndpointKind::Lemonade.locality(), Locality::Local);
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
        assert_eq!(e.locality, Some(Locality::Cloud));
        let e: RosterEntry = toml::from_str("id = \"x\"").unwrap();
        assert_eq!(e.locality, None);
        assert_eq!(e.priority, None);
        assert_eq!(e.strengths, None);
    }

    fn local_default() -> DefaultEndpoint {
        DefaultEndpoint {
            name: EndpointRef::new("main"),
            kind: EndpointKind::Ollama,
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
    fn a_hosted_openai_compatible_endpoint_is_cloud() {
        let groq =
            endpoint("kind = \"openai_compat\"\nbase_url = \"https://api.groq.com/openai/v1\"");
        assert_eq!(groq.effective_locality(), Locality::Cloud);
        // A local kind's own locality is not the endpoint's.
        assert_eq!(groq.kind.locality(), Locality::Local);
    }

    #[test]
    fn local_addresses_keep_non_cloud_kinds_local() {
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
    fn an_explicit_endpoint_locality_marks_a_public_name_local_but_never_a_cloud_kind() {
        let lan = endpoint(
            "kind = \"ollama\"\nbase_url = \"http://gpu.example.com:11434\"\nlocality = \"local\"",
        );
        assert_eq!(lan.effective_locality(), Locality::Local);
        let forced = endpoint(
            "kind = \"ollama\"\nbase_url = \"http://localhost:11434\"\nlocality = \"cloud\"",
        );
        assert_eq!(forced.effective_locality(), Locality::Cloud);
        for kind in ["anthropic", "openai"] {
            let ep = endpoint(&format!(
                "kind = \"{kind}\"\nbase_url = \"http://127.0.0.1:9\"\nlocality = \"local\""
            ));
            assert_eq!(ep.effective_locality(), Locality::Cloud, "{kind}");
        }
    }

    #[test]
    fn a_non_cloud_kind_with_no_address_fails_closed() {
        assert_eq!(
            endpoint("kind = \"openai_compat\"").effective_locality(),
            Locality::Cloud
        );
    }

    #[test]
    fn validate_explains_locality_surprises() {
        let c = toml::from_str::<Doc>(
            r#"
[routing.endpoints.claude]
kind = "anthropic"
locality = "local"

[routing.endpoints.groq]
kind = "openai_compat"
base_url = "https://api.groq.com/openai/v1"

[routing.endpoints.lan]
kind = "ollama"
base_url = "http://gpu.example.com:11434"
locality = "local"

[routing.endpoints.home]
kind = "ollama"
base_url = "http://192.168.1.20:11434"
"#,
        )
        .unwrap()
        .routing;
        let issues = c.validate(&local_default());
        assert_eq!(
            issues,
            vec![
                ConfigIssue::LocalOverrideIgnored {
                    endpoint: "claude".into(),
                    model: None
                },
                ConfigIssue::NonLocalAddress {
                    endpoint: "groq".into(),
                    host: "api.groq.com".into()
                },
            ]
        );
        let text: Vec<String> = issues.iter().map(ToString::to_string).collect();
        assert!(text[0].contains("cloud endpoint"), "{}", text[0]);
        assert!(
            text[1].contains("`api.groq.com`") && text[1].contains("locality = \"local\""),
            "{}",
            text[1]
        );
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
