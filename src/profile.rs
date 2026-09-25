//! What a candidate model is: its capabilities, size tier, strengths, and
//! where those facts came from.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

/// A technical capability a model has. Names match Ollama's `/api/show`
/// `capabilities` strings where one exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Completion,
    Tools,
    Vision,
    Thinking,
    Audio,
    Embedding,
}

impl Capability {
    /// Every capability, in declaration order.
    pub const ALL: [Capability; 6] = [
        Capability::Completion,
        Capability::Tools,
        Capability::Vision,
        Capability::Thinking,
        Capability::Audio,
        Capability::Embedding,
    ];
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Capability::Completion => "completion",
            Capability::Tools => "tools",
            Capability::Vision => "vision",
            Capability::Thinking => "thinking",
            Capability::Audio => "audio",
            Capability::Embedding => "embedding",
        })
    }
}

pub type CapabilitySet = BTreeSet<Capability>;

/// Coarse size/quality class. Operator-declared; discovery cannot know it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Small,
    Medium,
    Large,
}

impl Tier {
    /// 0 = small, 1 = medium, 2 = large.
    pub fn rank(self) -> i32 {
        match self {
            Tier::Small => 0,
            Tier::Medium => 1,
            Tier::Large => 2,
        }
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Tier::Small => "small",
            Tier::Medium => "medium",
            Tier::Large => "large",
        })
    }
}

/// What a model is good at. Operator-declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strength {
    Code,
    Reasoning,
    Chat,
    Summarize,
}

impl fmt::Display for Strength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Strength::Code => "code",
            Strength::Reasoning => "reasoning",
            Strength::Chat => "chat",
            Strength::Summarize => "summarize",
        })
    }
}

/// Whether requests to this model leave the machine.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Locality {
    #[default]
    Local,
    Cloud,
}

/// Whether the model can currently be routed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Availability {
    /// Seen by discovery.
    Available,
    /// Declared in the roster but not confirmed by discovery (not probed,
    /// or probed and absent). Still selectable; the fallback chain covers
    /// a failure.
    Unverified,
    /// Its endpoint was unreachable. Never selected.
    Unavailable,
}

/// Names an endpoint: a key of `[routing.endpoints]`, or the product's own
/// default backend.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EndpointRef(pub String);

impl EndpointRef {
    pub fn new(name: impl Into<String>) -> Self {
        EndpointRef(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for EndpointRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identifies one model: the same id on two endpoints is two models.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ModelKey {
    pub endpoint: EndpointRef,
    pub id: String,
}

impl fmt::Display for ModelKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.id, self.endpoint)
    }
}

/// Where a profile's facts came from, for `explain`-style output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProfileSource {
    pub discovered: bool,
    pub in_roster: bool,
}

/// One candidate model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProfile {
    pub id: String,
    pub endpoint: EndpointRef,
    pub locality: Locality,
    pub capabilities: CapabilitySet,
    /// Capabilities whose presence is unknown (the source couldn't say).
    /// A hard need is met by one of these, but ranks below a known match.
    pub unknown_capabilities: CapabilitySet,
    /// `None` = unknown (neither discovered nor declared).
    pub context_window: Option<u32>,
    pub tier: Tier,
    pub strengths: BTreeSet<Strength>,
    /// Operator tie-break; higher wins.
    pub priority: i32,
    pub availability: Availability,
    pub source: ProfileSource,
}

impl ModelProfile {
    pub fn new(id: impl Into<String>, endpoint: EndpointRef) -> Self {
        ModelProfile {
            id: id.into(),
            endpoint,
            locality: Locality::Local,
            capabilities: CapabilitySet::new(),
            unknown_capabilities: CapabilitySet::new(),
            context_window: None,
            tier: Tier::Medium,
            strengths: BTreeSet::new(),
            priority: 0,
            availability: Availability::Available,
            source: ProfileSource::default(),
        }
    }

    pub fn key(&self) -> ModelKey {
        ModelKey {
            endpoint: self.endpoint.clone(),
            id: self.id.clone(),
        }
    }

    /// Has `Embedding` but not `Completion`: it can only embed, so it must
    /// never win a generation request.
    pub fn is_embedding_only(&self) -> bool {
        self.capabilities.contains(&Capability::Embedding)
            && !self.capabilities.contains(&Capability::Completion)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_profile_has_neutral_defaults() {
        let p = ModelProfile::new("qwen3:8b", EndpointRef::new("ollama-main"));
        assert_eq!(p.id, "qwen3:8b");
        assert_eq!(p.endpoint.as_str(), "ollama-main");
        assert_eq!(p.locality, Locality::Local);
        assert!(p.capabilities.is_empty());
        assert_eq!(p.context_window, None);
        assert_eq!(p.tier, Tier::Medium);
        assert!(p.strengths.is_empty());
        assert_eq!(p.priority, 0);
        assert_eq!(p.availability, Availability::Available);
        assert_eq!(p.source, ProfileSource::default());
    }

    #[test]
    fn embedding_only_means_embedding_without_completion() {
        let mut p = ModelProfile::new("nomic-embed-text", EndpointRef::new("e"));
        p.capabilities.insert(Capability::Embedding);
        assert!(p.is_embedding_only());
        p.capabilities.insert(Capability::Completion);
        assert!(!p.is_embedding_only());
        let unknown = ModelProfile::new("unknown", EndpointRef::new("e"));
        assert!(
            !unknown.is_embedding_only(),
            "a model with no capability info must not look embedding-only"
        );
    }

    #[test]
    fn new_profile_has_no_unknown_capabilities() {
        let p = ModelProfile::new("m", EndpointRef::new("e"));
        assert!(p.unknown_capabilities.is_empty());
    }

    #[test]
    fn capability_displays_lowercase_and_all_lists_six() {
        let names: Vec<String> = Capability::ALL.iter().map(ToString::to_string).collect();
        assert_eq!(
            names,
            vec![
                "completion",
                "tools",
                "vision",
                "thinking",
                "audio",
                "embedding"
            ]
        );
    }

    #[test]
    fn tier_ranks_and_displays() {
        assert!(Tier::Small.rank() < Tier::Medium.rank());
        assert!(Tier::Medium.rank() < Tier::Large.rank());
        assert_eq!(Tier::Large.to_string(), "large");
        assert_eq!(Strength::Summarize.to_string(), "summarize");
        assert_eq!(EndpointRef::new("x").to_string(), "x");
    }

    #[test]
    fn enums_use_snake_case_on_the_wire() {
        assert_eq!(
            serde_json::to_string(&Capability::Vision).unwrap(),
            "\"vision\""
        );
        assert_eq!(
            serde_json::to_string(&Locality::Cloud).unwrap(),
            "\"cloud\""
        );
        let t: Tier = serde_json::from_str("\"small\"").unwrap();
        assert_eq!(t, Tier::Small);
        let e: EndpointRef = serde_json::from_str("\"ollama-main\"").unwrap();
        assert_eq!(e, EndpointRef::new("ollama-main"));
    }

    #[test]
    fn a_profile_is_identified_by_endpoint_and_id() {
        let p = ModelProfile::new("qwen3:8b", EndpointRef::new("gpu"));
        let k = p.key();
        assert_eq!(
            k,
            ModelKey {
                endpoint: EndpointRef::new("gpu"),
                id: "qwen3:8b".into()
            }
        );
        assert_eq!(k.to_string(), "qwen3:8b@gpu");
    }
}
