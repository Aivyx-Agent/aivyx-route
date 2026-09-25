//! The `[routing]` config section. Both products embed this shape.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use crate::profile::{CapabilitySet, Locality, Strength, Tier};
use crate::requirements::TaskOverrides;

/// The `[routing]` section. Unknown keys are ignored so each product can
/// add its own sub-tables (e.g. `aivyx-pa`'s `[routing.escalation]`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct RoutingConfig {
    /// `false` (the default) means today's single-model behavior.
    pub enabled: bool,
    pub discover: bool,
    pub endpoints: BTreeMap<String, EndpointConfig>,
    pub models: Vec<RosterEntry>,
    pub tasks: TaskOverrides,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        RoutingConfig {
            enabled: false,
            discover: true,
            endpoints: BTreeMap::new(),
            models: Vec::new(),
            tasks: TaskOverrides::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointKind {
    Ollama,
    /// `llama-server` in router mode (multi-model, `GET /models`).
    LlamaRouter,
    /// Anything serving `GET /v1/models` (single-model llama-server, Jan,
    /// vLLM, …).
    OpenaiCompat,
    Anthropic,
    Openai,
}

impl EndpointKind {
    pub fn locality(self) -> Locality {
        match self {
            EndpointKind::Anthropic | EndpointKind::Openai => Locality::Cloud,
            EndpointKind::Ollama | EndpointKind::LlamaRouter | EndpointKind::OpenaiCompat => {
                Locality::Local
            }
        }
    }

    pub fn default_base_url(self) -> Option<&'static str> {
        match self {
            EndpointKind::Ollama => Some("http://localhost:11434"),
            EndpointKind::LlamaRouter => Some("http://localhost:8080"),
            EndpointKind::OpenaiCompat | EndpointKind::Anthropic | EndpointKind::Openai => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointConfig {
    pub kind: EndpointKind,
    #[serde(default)]
    pub base_url: Option<String>,
}

impl EndpointConfig {
    /// The configured URL, else the kind's default.
    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref().or(self.kind.default_base_url())
    }
}

/// One `[[routing.models]]` entry.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RosterEntry {
    pub id: String,
    /// Key of `[routing.endpoints]`; `None` = the product's default backend.
    #[serde(default)]
    pub endpoint: Option<String>,
    /// Overrides the locality implied by the endpoint kind.
    #[serde(default)]
    pub locality: Option<Locality>,
    #[serde(default)]
    pub tier: Option<Tier>,
    #[serde(default)]
    pub strengths: BTreeSet<Strength>,
    #[serde(default)]
    pub priority: i32,
    /// Added to what discovery found.
    #[serde(default)]
    pub capabilities: CapabilitySet,
    /// Removed from what discovery found (e.g. unreliable tool calling).
    #[serde(default)]
    pub capabilities_deny: CapabilitySet,
    #[serde(default)]
    pub context_window: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::Capability;
    use serde::Deserialize;
    use std::collections::BTreeSet;

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
            BTreeSet::from([Strength::Code, Strength::Reasoning])
        );
        assert_eq!(qwen.priority, 10);
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
    fn roster_entry_locality_is_explicit_and_optional() {
        let e: RosterEntry = toml::from_str("id = \"claude\"\nlocality = \"cloud\"").unwrap();
        assert_eq!(e.locality, Some(Locality::Cloud));
        let e: RosterEntry = toml::from_str("id = \"x\"").unwrap();
        assert_eq!(e.locality, None);
        assert_eq!(e.priority, 0);
    }
}
