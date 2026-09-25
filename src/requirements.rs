//! What a request needs (hard filters) and prefers (soft scoring), and the
//! task kinds call sites tag their requests with.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::Deserialize;

use crate::profile::{Strength, Tier};

/// One hard requirement; a model failing any is never selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HardNeed {
    Vision,
    Tools,
    Audio,
    Embedding,
    Context(u32),
}

impl fmt::Display for HardNeed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HardNeed::Vision => f.write_str("vision"),
            HardNeed::Tools => f.write_str("tool calling"),
            HardNeed::Audio => f.write_str("audio input"),
            HardNeed::Embedding => f.write_str("embeddings"),
            HardNeed::Context(n) => write!(f, "a context window of at least {n} tokens"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HardRequirements {
    pub vision: bool,
    pub tools: bool,
    pub audio: bool,
    pub embedding: bool,
    pub min_context: Option<u32>,
}

impl HardRequirements {
    /// The active needs, in a fixed order.
    pub fn needs(&self) -> Vec<HardNeed> {
        let mut out = Vec::new();
        if self.vision {
            out.push(HardNeed::Vision);
        }
        if self.tools {
            out.push(HardNeed::Tools);
        }
        if self.audio {
            out.push(HardNeed::Audio);
        }
        if self.embedding {
            out.push(HardNeed::Embedding);
        }
        if let Some(n) = self.min_context {
            out.push(HardNeed::Context(n));
        }
        out
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SoftPreferences {
    pub tier: Option<Tier>,
    pub strengths: BTreeSet<Strength>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Requirements {
    pub hard: HardRequirements,
    pub soft: SoftPreferences,
}

impl Requirements {
    pub fn builder() -> RequirementsBuilder {
        RequirementsBuilder::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct RequirementsBuilder {
    req: Requirements,
}

impl RequirementsBuilder {
    pub fn vision(mut self) -> Self {
        self.req.hard.vision = true;
        self
    }

    pub fn tools(mut self) -> Self {
        self.req.hard.tools = true;
        self
    }

    pub fn audio(mut self) -> Self {
        self.req.hard.audio = true;
        self
    }

    /// Keeps the largest value if called more than once.
    pub fn min_context(mut self, tokens: u32) -> Self {
        self.req.hard.min_context =
            Some(self.req.hard.min_context.map_or(tokens, |t| t.max(tokens)));
        self
    }

    /// Sets the soft preferences for `task` (defaults + operator overrides).
    /// `TaskKind::Embed` also sets the hard `embedding` need.
    pub fn task(mut self, task: &TaskKind, overrides: &TaskOverrides) -> Self {
        self.req.soft = overrides.preferences_for(task);
        if matches!(task, TaskKind::Embed) {
            self.req.hard.embedding = true;
        }
        self
    }

    pub fn build(self) -> Requirements {
        self.req
    }
}

/// What a call site is doing. Maps to a default tier and strengths.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TaskKind {
    Chat,
    CodeEdit,
    Plan,
    Judge,
    Summarize,
    Compact,
    Classify,
    Embed,
    Custom(String),
}

impl TaskKind {
    /// The key used in `[routing.tasks]`.
    pub fn name(&self) -> &str {
        match self {
            TaskKind::Chat => "chat",
            TaskKind::CodeEdit => "code_edit",
            TaskKind::Plan => "plan",
            TaskKind::Judge => "judge",
            TaskKind::Summarize => "summarize",
            TaskKind::Compact => "compact",
            TaskKind::Classify => "classify",
            TaskKind::Embed => "embed",
            TaskKind::Custom(name) => name,
        }
    }

    pub fn default_preferences(&self) -> SoftPreferences {
        let (tier, strengths): (Option<Tier>, &[Strength]) = match self {
            TaskKind::Chat => (Some(Tier::Medium), &[Strength::Chat]),
            TaskKind::CodeEdit => (Some(Tier::Large), &[Strength::Code]),
            TaskKind::Plan => (Some(Tier::Large), &[Strength::Reasoning]),
            TaskKind::Judge => (Some(Tier::Medium), &[Strength::Reasoning]),
            TaskKind::Summarize | TaskKind::Compact => (Some(Tier::Small), &[Strength::Summarize]),
            TaskKind::Classify => (Some(Tier::Small), &[]),
            TaskKind::Embed => (None, &[]),
            TaskKind::Custom(_) => (Some(Tier::Medium), &[]),
        };
        SoftPreferences {
            tier,
            strengths: strengths.iter().copied().collect(),
        }
    }

    /// Main-thread kinds keep the conversation's current model; every other
    /// kind is a side call routed freely.
    pub fn is_sticky(&self) -> bool {
        matches!(self, TaskKind::Chat | TaskKind::CodeEdit)
    }
}

/// One `[routing.tasks]` entry. Unset fields keep the task's default.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskOverride {
    #[serde(default)]
    pub tier: Option<Tier>,
    #[serde(default)]
    pub strengths: Option<BTreeSet<Strength>>,
}

/// The `[routing.tasks]` table, keyed by [`TaskKind::name`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(transparent)]
pub struct TaskOverrides(pub BTreeMap<String, TaskOverride>);

impl TaskOverrides {
    pub fn preferences_for(&self, task: &TaskKind) -> SoftPreferences {
        let mut prefs = task.default_preferences();
        if let Some(o) = self.0.get(task.name()) {
            if let Some(tier) = o.tier {
                prefs.tier = Some(tier);
            }
            if let Some(strengths) = &o.strengths {
                prefs.strengths = strengths.clone();
            }
        }
        prefs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_collects_hard_needs_in_fixed_order() {
        let r = Requirements::builder()
            .tools()
            .vision()
            .min_context(32_000)
            .build();
        assert_eq!(
            r.hard.needs(),
            vec![HardNeed::Vision, HardNeed::Tools, HardNeed::Context(32_000)]
        );
    }

    #[test]
    fn min_context_keeps_the_largest_value() {
        let r = Requirements::builder()
            .min_context(8_000)
            .min_context(4_000)
            .build();
        assert_eq!(r.hard.min_context, Some(8_000));
    }

    #[test]
    fn task_defaults_match_the_spec_table() {
        let cases: Vec<(TaskKind, Option<Tier>, Vec<Strength>)> = vec![
            (TaskKind::Chat, Some(Tier::Medium), vec![Strength::Chat]),
            (TaskKind::CodeEdit, Some(Tier::Large), vec![Strength::Code]),
            (TaskKind::Plan, Some(Tier::Large), vec![Strength::Reasoning]),
            (
                TaskKind::Judge,
                Some(Tier::Medium),
                vec![Strength::Reasoning],
            ),
            (
                TaskKind::Summarize,
                Some(Tier::Small),
                vec![Strength::Summarize],
            ),
            (
                TaskKind::Compact,
                Some(Tier::Small),
                vec![Strength::Summarize],
            ),
            (TaskKind::Classify, Some(Tier::Small), vec![]),
            (TaskKind::Embed, None, vec![]),
            (
                TaskKind::Custom("reviewer".into()),
                Some(Tier::Medium),
                vec![],
            ),
        ];
        for (task, tier, strengths) in cases {
            let p = task.default_preferences();
            assert_eq!(p.tier, tier, "{task:?}");
            assert_eq!(
                p.strengths,
                strengths.into_iter().collect::<BTreeSet<_>>(),
                "{task:?}"
            );
        }
    }

    #[test]
    fn embed_task_sets_the_hard_embedding_need() {
        let r = Requirements::builder()
            .task(&TaskKind::Embed, &TaskOverrides::default())
            .build();
        assert!(r.hard.embedding);
        assert_eq!(r.hard.needs(), vec![HardNeed::Embedding]);
    }

    #[test]
    fn task_sets_soft_preferences() {
        let r = Requirements::builder()
            .task(&TaskKind::Plan, &TaskOverrides::default())
            .build();
        assert_eq!(r.soft.tier, Some(Tier::Large));
        assert!(r.hard.needs().is_empty());
    }

    #[test]
    fn overrides_replace_only_the_fields_they_set() {
        let mut o = TaskOverrides::default();
        o.0.insert(
            "summarize".into(),
            TaskOverride {
                tier: Some(Tier::Medium),
                strengths: None,
            },
        );
        let p = o.preferences_for(&TaskKind::Summarize);
        assert_eq!(p.tier, Some(Tier::Medium));
        assert_eq!(p.strengths, BTreeSet::from([Strength::Summarize]));
    }

    #[test]
    fn custom_task_overrides_are_keyed_by_name() {
        let mut o = TaskOverrides::default();
        o.0.insert(
            "reviewer".into(),
            TaskOverride {
                tier: Some(Tier::Large),
                strengths: Some(BTreeSet::from([Strength::Code])),
            },
        );
        let p = o.preferences_for(&TaskKind::Custom("reviewer".into()));
        assert_eq!(p.tier, Some(Tier::Large));
        assert_eq!(p.strengths, BTreeSet::from([Strength::Code]));
    }

    #[test]
    fn only_chat_and_code_edit_are_sticky() {
        assert!(TaskKind::Chat.is_sticky());
        assert!(TaskKind::CodeEdit.is_sticky());
        for k in [
            TaskKind::Plan,
            TaskKind::Judge,
            TaskKind::Summarize,
            TaskKind::Compact,
            TaskKind::Classify,
            TaskKind::Embed,
            TaskKind::Custom("x".into()),
        ] {
            assert!(!k.is_sticky(), "{k:?}");
        }
    }

    #[test]
    fn names_and_display() {
        assert_eq!(TaskKind::CodeEdit.name(), "code_edit");
        assert_eq!(TaskKind::Custom("reviewer".into()).name(), "reviewer");
        assert_eq!(HardNeed::Vision.to_string(), "vision");
        assert_eq!(HardNeed::Tools.to_string(), "tool calling");
        assert_eq!(
            HardNeed::Context(32_000).to_string(),
            "a context window of at least 32000 tokens"
        );
    }

    #[test]
    fn task_overrides_deserialize_from_toml() {
        let o: TaskOverrides = toml::from_str(
            "summarize = { tier = \"medium\" }\nplan = { strengths = [\"code\"] }\n",
        )
        .unwrap();
        assert_eq!(o.0["summarize"].tier, Some(Tier::Medium));
        assert_eq!(
            o.0["plan"].strengths,
            Some(BTreeSet::from([Strength::Code]))
        );
        assert!(toml::from_str::<TaskOverrides>("x = { teir = \"small\" }").is_err());
    }
}
