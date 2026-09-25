//! Choosing one model for one request: hard filters, then deterministic
//! ranking, with a human-readable reason.

use std::cmp::Reverse;
use std::fmt;

use serde::Serialize;

use crate::profile::{
    Availability, Capability, CapabilitySet, EndpointRef, Locality, ModelKey, ModelProfile,
    Strength, Tier,
};
use crate::requirements::{HardNeed, Requirements};

/// Constraints the product imposes on this call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Policy {
    /// The conversation's current model. Kept if it meets every hard need.
    pub sticky_model: Option<ModelKey>,
    /// Cloud profiles are candidates only when `true`.
    pub allow_cloud: bool,
    /// Models never to select.
    pub exclude: Vec<ModelKey>,
}

/// One clause of a decision's explanation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasonPart {
    Required(HardNeed),
    Sticky,
    OnlyCandidate,
    TierFit {
        wanted: Tier,
        got: Tier,
    },
    Strengths(Vec<Strength>),
    Priority(i32),
    UnknownContextWindow,
    /// Hard needs met only because the capability's presence is unknown.
    UnverifiedCapabilities(Vec<Capability>),
}

impl fmt::Display for ReasonPart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReasonPart::Required(need) => write!(f, "{need} required"),
            ReasonPart::Sticky => f.write_str("kept the conversation's current model"),
            ReasonPart::OnlyCandidate => f.write_str("only model meeting the hard requirements"),
            ReasonPart::TierFit { wanted, got } if wanted == got => {
                write!(f, "matches the {wanted} tier wanted")
            }
            ReasonPart::TierFit { wanted, got } => {
                write!(f, "closest to the {wanted} tier wanted (is {got})")
            }
            ReasonPart::Strengths(strengths) => {
                let names: Vec<String> = strengths.iter().map(ToString::to_string).collect();
                write!(f, "matches strengths: {}", names.join(", "))
            }
            ReasonPart::Priority(p) => write!(f, "operator priority {p}"),
            ReasonPart::UnknownContextWindow => f.write_str(
                "context window unknown; no model with a known, large-enough window qualified",
            ),
            ReasonPart::UnverifiedCapabilities(caps) => {
                let names: Vec<String> = caps.iter().map(ToString::to_string).collect();
                write!(f, "assumed but unverified: {}", names.join(", "))
            }
        }
    }
}

/// The chosen model, ranked fallbacks, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Decision {
    pub model: ModelProfile,
    pub fallbacks: Vec<ModelProfile>,
    pub reason: Vec<ReasonPart>,
}

impl fmt::Display for Decision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "chose `{}`", self.model.id)?;
        if !self.reason.is_empty() {
            let parts: Vec<String> = self.reason.iter().map(ToString::to_string).collect();
            write!(f, ": {}", parts.join("; "))?;
        }
        Ok(())
    }
}

/// A hard need no eligible model can meet (given the other needs).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Unmet {
    pub need: HardNeed,
    /// Best-ranked model that meets every *other* need but not this one.
    pub near_miss: Option<String>,
}

/// No eligible model meets every hard need.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NoCandidate {
    pub unmet: Vec<Unmet>,
    /// How many models survived the eligibility filter.
    pub considered: usize,
}

impl fmt::Display for NoCandidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.considered == 0 {
            f.write_str("no candidate models are available")?;
            if !self.unmet.is_empty() {
                let needs: Vec<String> = self.unmet.iter().map(|u| u.need.to_string()).collect();
                write!(f, " (needed: {})", needs.join(", "))?;
            }
            return Ok(());
        }
        let parts: Vec<String> = self
            .unmet
            .iter()
            .map(|u| match &u.near_miss {
                Some(id) => format!("{} (closest: `{id}`)", u.need),
                None => u.need.to_string(),
            })
            .collect();
        write!(
            f,
            "no model meets the hard requirements: {}",
            parts.join(", ")
        )
    }
}

impl std::error::Error for NoCandidate {}

/// Pick one model for `req` from `candidates`. Pure and deterministic.
pub fn select(
    req: &Requirements,
    candidates: &[ModelProfile],
    policy: &Policy,
) -> Result<Decision, NoCandidate> {
    let eligible: Vec<&ModelProfile> = candidates
        .iter()
        .filter(|p| is_eligible(p, req, policy))
        .collect();
    let needs = req.hard.needs();
    let passing: Vec<&ModelProfile> = eligible
        .iter()
        .copied()
        .filter(|p| needs.iter().all(|n| satisfies(p, *n)))
        .collect();
    if passing.is_empty() {
        return Err(no_candidate(&eligible, &needs, req));
    }

    let mut ranked = rank(passing, req);
    let mut reason: Vec<ReasonPart> = needs.iter().map(|n| ReasonPart::Required(*n)).collect();
    let sticky_pos = policy
        .sticky_model
        .as_ref()
        .and_then(|k| ranked.iter().position(|p| p.key() == *k));
    let model = match sticky_pos {
        Some(pos) => {
            reason.push(ReasonPart::Sticky);
            ranked.remove(pos)
        }
        None => {
            let model = ranked.remove(0);
            explain_rank(model, &ranked, req, &mut reason);
            // Ranking put known windows first, so this one won only because
            // none qualified. Not true of a kept sticky model.
            if req.hard.min_context.is_some() && model.context_window.is_none() {
                reason.push(ReasonPart::UnknownContextWindow);
            }
            model
        }
    };
    let assumed = assumed_capabilities(model, &needs);
    if !assumed.is_empty() {
        reason.push(ReasonPart::UnverifiedCapabilities(assumed));
    }
    Ok(Decision {
        model: model.clone(),
        fallbacks: ranked.into_iter().cloned().collect(),
        reason,
    })
}

/// Hard needs `profile` definitely fails. Unknown capabilities and an
/// unknown context window count as met, as in [`select`]. For validating an
/// explicit operator pin.
pub fn unmet_needs(req: &Requirements, profile: &ModelProfile) -> Vec<HardNeed> {
    req.hard
        .needs()
        .into_iter()
        .filter(|n| !satisfies(profile, *n))
        .collect()
}

/// The profile with this key, if any.
pub fn find<'a>(profiles: &'a [ModelProfile], key: &ModelKey) -> Option<&'a ModelProfile> {
    profiles
        .iter()
        .find(|p| p.endpoint == key.endpoint && p.id == key.id)
}

fn is_eligible(p: &ModelProfile, req: &Requirements, policy: &Policy) -> bool {
    p.availability != Availability::Unavailable
        && !policy.exclude.contains(&p.key())
        && (policy.allow_cloud || p.locality == Locality::Local)
        && (req.hard.embedding || !p.is_embedding_only())
}

/// The capability a hard need asks for, if it is a capability need.
fn capability_for(need: HardNeed) -> Option<Capability> {
    match need {
        HardNeed::Vision => Some(Capability::Vision),
        HardNeed::Tools => Some(Capability::Tools),
        HardNeed::Audio => Some(Capability::Audio),
        HardNeed::Embedding => Some(Capability::Embedding),
        HardNeed::Context(_) => None,
    }
}

/// Unknown capabilities and an unknown context window count as met.
fn satisfies(p: &ModelProfile, need: HardNeed) -> bool {
    match (need, capability_for(need)) {
        (HardNeed::Context(min), _) => p.context_window.is_none_or(|w| w >= min),
        (_, cap) => {
            cap.is_some_and(|c| p.capabilities.contains(&c) || p.unknown_capabilities.contains(&c))
        }
    }
}

/// Capabilities among `needs` that `p` meets only via `unknown_capabilities`,
/// in `Capability` order.
fn assumed_capabilities(p: &ModelProfile, needs: &[HardNeed]) -> Vec<Capability> {
    let assumed: CapabilitySet = needs
        .iter()
        .filter_map(|n| capability_for(*n))
        .filter(|c| !p.capabilities.contains(c) && p.unknown_capabilities.contains(c))
        .collect();
    assumed.into_iter().collect()
}

fn tier_penalty(wanted: Option<Tier>, got: Tier) -> u8 {
    let Some(wanted) = wanted else {
        return 0;
    };
    match got.rank() - wanted.rank() {
        0 => 0,
        1 => 1,
        2 => 2,
        -1 => 3,
        _ => 5,
    }
}

type RankKey = (
    bool,
    bool,
    u8,
    Reverse<usize>,
    Reverse<i32>,
    EndpointRef,
    String,
);

fn rank_key(p: &ModelProfile, req: &Requirements) -> RankKey {
    let relies_on_unknown_caps = !assumed_capabilities(p, &req.hard.needs()).is_empty();
    let unknown_ctx = req.hard.min_context.is_some() && p.context_window.is_none();
    let overlap = p.strengths.intersection(&req.soft.strengths).count();
    (
        relies_on_unknown_caps,
        unknown_ctx,
        tier_penalty(req.soft.tier, p.tier),
        Reverse(overlap),
        Reverse(p.priority),
        p.endpoint.clone(),
        p.id.clone(),
    )
}

fn rank<'a>(mut profiles: Vec<&'a ModelProfile>, req: &Requirements) -> Vec<&'a ModelProfile> {
    profiles.sort_by_cached_key(|p| rank_key(p, req));
    profiles
}

fn explain_rank(
    model: &ModelProfile,
    rest: &[&ModelProfile],
    req: &Requirements,
    reason: &mut Vec<ReasonPart>,
) {
    if rest.is_empty() {
        reason.push(ReasonPart::OnlyCandidate);
        return;
    }
    if let Some(wanted) = req.soft.tier {
        reason.push(ReasonPart::TierFit {
            wanted,
            got: model.tier,
        });
    }
    let matched: Vec<Strength> = model
        .strengths
        .intersection(&req.soft.strengths)
        .copied()
        .collect();
    if !matched.is_empty() {
        reason.push(ReasonPart::Strengths(matched));
    }
    if model.priority != 0 {
        reason.push(ReasonPart::Priority(model.priority));
    }
}

fn no_candidate(eligible: &[&ModelProfile], needs: &[HardNeed], req: &Requirements) -> NoCandidate {
    let unmet = needs
        .iter()
        .filter_map(|&need| {
            let others: Vec<&ModelProfile> = eligible
                .iter()
                .copied()
                .filter(|p| {
                    needs
                        .iter()
                        .filter(|n| **n != need)
                        .all(|n| satisfies(p, *n))
                })
                .collect();
            if others.iter().any(|p| satisfies(p, need)) {
                return None;
            }
            let near_miss = rank(others, req).first().map(|p| p.id.clone());
            Some(Unmet { need, near_miss })
        })
        .collect();
    NoCandidate {
        unmet,
        considered: eligible.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::requirements::{RequirementsBuilder, TaskKind, TaskOverrides};

    fn m(id: &str, tier: Tier) -> ModelProfile {
        let mut p = ModelProfile::new(id, EndpointRef::new("local"));
        p.tier = tier;
        p.capabilities.insert(Capability::Completion);
        p
    }

    fn key(id: &str) -> ModelKey {
        ModelKey {
            endpoint: EndpointRef::new("local"),
            id: id.into(),
        }
    }

    fn on(mut p: ModelProfile, endpoint: &str) -> ModelProfile {
        p.endpoint = EndpointRef::new(endpoint);
        p
    }

    fn with_caps(mut p: ModelProfile, caps: &[Capability]) -> ModelProfile {
        p.capabilities.extend(caps.iter().copied());
        p
    }

    fn with_strengths(mut p: ModelProfile, s: &[Strength]) -> ModelProfile {
        p.strengths.extend(s.iter().copied());
        p
    }

    fn for_task(task: TaskKind) -> RequirementsBuilder {
        Requirements::builder().task(&task, &TaskOverrides::default())
    }

    fn fallback_ids(d: &Decision) -> Vec<&str> {
        d.fallbacks.iter().map(|p| p.id.as_str()).collect()
    }

    #[test]
    fn vision_need_filters_to_vision_models() {
        let req = for_task(TaskKind::Chat).vision().build();
        let c = [
            m("text", Tier::Medium),
            with_caps(m("seer", Tier::Small), &[Capability::Vision]),
        ];
        let d = select(&req, &c, &Policy::default()).unwrap();
        assert_eq!(d.model.id, "seer");
        assert!(d.fallbacks.is_empty());
    }

    #[test]
    fn tools_need_filters_to_tool_models() {
        let req = for_task(TaskKind::Chat).tools().build();
        let c = [
            m("plain", Tier::Medium),
            with_caps(m("caller", Tier::Medium), &[Capability::Tools]),
        ];
        assert_eq!(
            select(&req, &c, &Policy::default()).unwrap().model.id,
            "caller"
        );
    }

    #[test]
    fn min_context_drops_small_windows_and_ranks_unknown_last() {
        let req = for_task(TaskKind::Chat).min_context(32_000).build();
        let mut small = m("a-small-ctx", Tier::Medium);
        small.context_window = Some(8_192);
        let mut big = m("b-big-ctx", Tier::Medium);
        big.context_window = Some(131_072);
        let unknown = m("c-unknown", Tier::Medium);
        let d = select(&req, &[small, big, unknown], &Policy::default()).unwrap();
        assert_eq!(d.model.id, "b-big-ctx");
        assert_eq!(fallback_ids(&d), vec!["c-unknown"]);
    }

    #[test]
    fn unknown_context_window_is_chosen_when_nothing_better_and_says_so() {
        let req = for_task(TaskKind::Chat).min_context(32_000).build();
        let mut small = m("a-small-ctx", Tier::Medium);
        small.context_window = Some(8_192);
        let d = select(
            &req,
            &[small, m("c-unknown", Tier::Medium)],
            &Policy::default(),
        )
        .unwrap();
        assert_eq!(d.model.id, "c-unknown");
        assert_eq!(
            d.reason,
            vec![
                ReasonPart::Required(HardNeed::Context(32_000)),
                ReasonPart::OnlyCandidate,
                ReasonPart::UnknownContextWindow,
            ]
        );
    }

    #[test]
    fn exact_tier_wins() {
        let req = for_task(TaskKind::Summarize).build();
        let d = select(
            &req,
            &[m("big", Tier::Large), m("tiny", Tier::Small)],
            &Policy::default(),
        )
        .unwrap();
        assert_eq!(d.model.id, "tiny");
    }

    #[test]
    fn larger_beats_smaller_when_no_exact_tier() {
        let req = for_task(TaskKind::Chat).build(); // wants Medium
        let d = select(
            &req,
            &[m("tiny", Tier::Small), m("big", Tier::Large)],
            &Policy::default(),
        )
        .unwrap();
        assert_eq!(d.model.id, "big");
    }

    #[test]
    fn one_step_short_beats_two_steps_short() {
        let req = for_task(TaskKind::Plan).build(); // wants Large
        let d = select(
            &req,
            &[m("tiny", Tier::Small), m("mid", Tier::Medium)],
            &Policy::default(),
        )
        .unwrap();
        assert_eq!(d.model.id, "mid");
    }

    #[test]
    fn strengths_beat_priority_within_a_tier() {
        let req = for_task(TaskKind::Plan).build(); // Large + Reasoning
        let mut favoured = m("favoured", Tier::Large);
        favoured.priority = 10;
        let thinker = with_strengths(m("thinker", Tier::Large), &[Strength::Reasoning]);
        let d = select(&req, &[favoured, thinker], &Policy::default()).unwrap();
        assert_eq!(d.model.id, "thinker");
    }

    #[test]
    fn priority_then_id_break_ties() {
        let req = for_task(TaskKind::Chat).build();
        let mut high = m("zeta", Tier::Medium);
        high.priority = 5;
        let d = select(
            &req,
            &[m("beta", Tier::Medium), high, m("alpha", Tier::Medium)],
            &Policy::default(),
        )
        .unwrap();
        assert_eq!(d.model.id, "zeta");
        assert_eq!(fallback_ids(&d), vec!["alpha", "beta"]);
    }

    #[test]
    fn sticky_model_is_kept_when_it_passes() {
        let req = for_task(TaskKind::Chat).build();
        let better = with_strengths(m("better", Tier::Medium), &[Strength::Chat]);
        let policy = Policy {
            sticky_model: Some(key("current")),
            ..Policy::default()
        };
        let d = select(&req, &[better, m("current", Tier::Small)], &policy).unwrap();
        assert_eq!(d.model.id, "current");
        assert_eq!(d.reason, vec![ReasonPart::Sticky]);
        assert_eq!(fallback_ids(&d), vec!["better"]);
    }

    #[test]
    fn sticky_model_is_replaced_when_it_fails_a_hard_need() {
        let req = for_task(TaskKind::Chat).vision().build();
        let policy = Policy {
            sticky_model: Some(key("current")),
            ..Policy::default()
        };
        let c = [
            m("current", Tier::Medium),
            with_caps(m("seer", Tier::Medium), &[Capability::Vision]),
        ];
        let d = select(&req, &c, &policy).unwrap();
        assert_eq!(d.model.id, "seer");
        assert!(!d.reason.contains(&ReasonPart::Sticky));
    }

    #[test]
    fn cloud_models_need_allow_cloud() {
        let req = for_task(TaskKind::Plan).build();
        let mut cloud = m("cloud-large", Tier::Large);
        cloud.locality = Locality::Cloud;
        let c = [cloud, m("local-mid", Tier::Medium)];
        assert_eq!(
            select(&req, &c, &Policy::default()).unwrap().model.id,
            "local-mid"
        );
        let allow = Policy {
            allow_cloud: true,
            ..Policy::default()
        };
        assert_eq!(select(&req, &c, &allow).unwrap().model.id, "cloud-large");
    }

    #[test]
    fn unavailable_is_skipped_unverified_is_not() {
        let req = for_task(TaskKind::Chat).build();
        let mut down = m("down", Tier::Medium);
        down.availability = Availability::Unavailable;
        let mut maybe = m("maybe", Tier::Small);
        maybe.availability = Availability::Unverified;
        assert_eq!(
            select(&req, &[down, maybe], &Policy::default())
                .unwrap()
                .model
                .id,
            "maybe"
        );
    }

    #[test]
    fn excluded_models_are_skipped() {
        let req = for_task(TaskKind::Chat).build();
        let policy = Policy {
            exclude: vec![key("banned")],
            ..Policy::default()
        };
        let d = select(
            &req,
            &[m("banned", Tier::Medium), m("ok", Tier::Small)],
            &policy,
        )
        .unwrap();
        assert_eq!(d.model.id, "ok");
    }

    #[test]
    fn embedding_only_models_never_win_generation() {
        let req = for_task(TaskKind::Chat).build();
        let mut embed = ModelProfile::new("nomic-embed-text", EndpointRef::new("local"));
        embed.capabilities.insert(Capability::Embedding);
        let d = select(
            &req,
            &[embed.clone(), m("chatty", Tier::Small)],
            &Policy::default(),
        )
        .unwrap();
        assert_eq!(d.model.id, "chatty");
        let err = select(&req, &[embed], &Policy::default()).unwrap_err();
        assert_eq!(err.considered, 0);
    }

    #[test]
    fn embed_task_requires_embedding_capability() {
        let req = for_task(TaskKind::Embed).build();
        let mut embed = ModelProfile::new("nomic-embed-text", EndpointRef::new("local"));
        embed.capabilities.insert(Capability::Embedding);
        let d = select(&req, &[m("chatty", Tier::Small), embed], &Policy::default()).unwrap();
        assert_eq!(d.model.id, "nomic-embed-text");
    }

    #[test]
    fn no_candidate_names_the_unmet_need_and_nearest_miss() {
        let req = for_task(TaskKind::Chat).vision().build();
        let err = select(
            &req,
            &[m("mid", Tier::Medium), m("big", Tier::Large)],
            &Policy::default(),
        )
        .unwrap_err();
        assert_eq!(err.considered, 2);
        assert_eq!(
            err.unmet,
            vec![Unmet {
                need: HardNeed::Vision,
                near_miss: Some("mid".into())
            }]
        );
        assert_eq!(
            err.to_string(),
            "no model meets the hard requirements: vision (closest: `mid`)"
        );
    }

    #[test]
    fn no_candidate_reports_each_need_when_no_model_has_both() {
        let req = for_task(TaskKind::Chat).vision().tools().build();
        let c = [
            with_caps(m("seer", Tier::Medium), &[Capability::Vision]),
            with_caps(m("caller", Tier::Medium), &[Capability::Tools]),
        ];
        let err = select(&req, &c, &Policy::default()).unwrap_err();
        assert_eq!(
            err.unmet,
            vec![
                Unmet {
                    need: HardNeed::Vision,
                    near_miss: Some("caller".into())
                },
                Unmet {
                    need: HardNeed::Tools,
                    near_miss: Some("seer".into())
                },
            ]
        );
    }

    #[test]
    fn empty_candidates() {
        let req = for_task(TaskKind::Chat).build();
        let err = select(&req, &[], &Policy::default()).unwrap_err();
        assert_eq!(err.considered, 0);
        assert_eq!(err.to_string(), "no candidate models are available");
    }

    #[test]
    fn decision_display_explains_the_choice() {
        let req = for_task(TaskKind::Summarize).build();
        let s = with_strengths(m("s", Tier::Small), &[Strength::Summarize]);
        let d = select(&req, &[m("l", Tier::Large), s], &Policy::default()).unwrap();
        assert_eq!(
            d.to_string(),
            "chose `s`: matches the small tier wanted; matches strengths: summarize"
        );

        let req = for_task(TaskKind::Chat).vision().build();
        let v = with_caps(m("v", Tier::Large), &[Capability::Vision]);
        let d = select(&req, &[v, m("t", Tier::Medium)], &Policy::default()).unwrap();
        assert_eq!(
            d.to_string(),
            "chose `v`: vision required; only model meeting the hard requirements"
        );
    }

    fn with_unknown(mut p: ModelProfile, caps: &[Capability]) -> ModelProfile {
        p.unknown_capabilities.extend(caps.iter().copied());
        p
    }

    #[test]
    fn unknown_capability_passes_a_hard_need_and_says_so() {
        let req = for_task(TaskKind::Chat).tools().vision().build();
        let c = [with_unknown(
            m("compat", Tier::Medium),
            &[Capability::Tools, Capability::Vision],
        )];
        let d = select(&req, &c, &Policy::default()).unwrap();
        assert_eq!(d.model.id, "compat");
        assert!(d.reason.contains(&ReasonPart::UnverifiedCapabilities(vec![
            Capability::Tools,
            Capability::Vision
        ])));
        assert!(
            d.to_string()
                .contains("assumed but unverified: tools, vision"),
            "{d}"
        );
    }

    #[test]
    fn known_capability_outranks_unknown_even_at_a_worse_tier() {
        let req = for_task(TaskKind::Chat).tools().build(); // wants Medium
        let assumed = with_unknown(m("assumed", Tier::Medium), &[Capability::Tools]);
        let known = with_caps(m("known", Tier::Small), &[Capability::Tools]);
        let d = select(&req, &[assumed, known], &Policy::default()).unwrap();
        assert_eq!(d.model.id, "known");
        assert_eq!(fallback_ids(&d), vec!["assumed"]);
        assert!(
            !d.reason
                .iter()
                .any(|r| matches!(r, ReasonPart::UnverifiedCapabilities(_)))
        );
    }

    #[test]
    fn unknown_other_capabilities_do_not_satisfy_vision() {
        // llama router, text-only modalities: vision is known absent.
        let req = for_task(TaskKind::Chat).vision().build();
        let router = with_unknown(
            m("router", Tier::Medium),
            &[
                Capability::Completion,
                Capability::Tools,
                Capability::Thinking,
                Capability::Embedding,
            ],
        );
        let err = select(&req, &[router], &Policy::default()).unwrap_err();
        assert_eq!(err.unmet[0].need, HardNeed::Vision);
    }

    #[test]
    fn near_miss_logic_counts_unknown_as_met() {
        // `assumed` meets tools only via unknown, so it is the near-miss
        // for vision (it would be none if unknown counted as absent).
        let req = for_task(TaskKind::Chat).tools().vision().build();
        let assumed = with_unknown(m("assumed", Tier::Medium), &[Capability::Tools]);
        let err = select(&req, &[assumed], &Policy::default()).unwrap_err();
        assert_eq!(
            err.unmet,
            vec![
                Unmet {
                    need: HardNeed::Vision,
                    near_miss: Some("assumed".into())
                },
                Unmet {
                    need: HardNeed::Tools,
                    near_miss: None
                },
            ]
        );
    }

    #[test]
    fn deterministic_regardless_of_input_order() {
        let req = for_task(TaskKind::Chat).build();
        let a = [
            m("b", Tier::Medium),
            m("a", Tier::Medium),
            m("c", Tier::Small),
        ];
        let b = [
            m("c", Tier::Small),
            m("a", Tier::Medium),
            m("b", Tier::Medium),
        ];
        assert_eq!(
            select(&req, &a, &Policy::default()).unwrap(),
            select(&req, &b, &Policy::default()).unwrap()
        );
    }

    #[test]
    fn sticky_keeps_the_right_endpoint_for_a_shared_id() {
        let req = for_task(TaskKind::Chat).build();
        let c = [
            on(m("qwen3:8b", Tier::Medium), "a-gpu"),
            on(m("qwen3:8b", Tier::Medium), "b-gpu"),
        ];
        let policy = Policy {
            sticky_model: Some(ModelKey {
                endpoint: EndpointRef::new("b-gpu"),
                id: "qwen3:8b".into(),
            }),
            ..Policy::default()
        };
        let d = select(&req, &c, &policy).unwrap();
        assert_eq!(d.model.endpoint, EndpointRef::new("b-gpu"));
        assert_eq!(d.reason, vec![ReasonPart::Sticky]);
    }

    #[test]
    fn excluding_one_endpoints_copy_leaves_the_other() {
        let req = for_task(TaskKind::Chat).build();
        let c = [
            on(m("qwen3:8b", Tier::Medium), "a-gpu"),
            on(m("qwen3:8b", Tier::Medium), "b-gpu"),
        ];
        let policy = Policy {
            exclude: vec![ModelKey {
                endpoint: EndpointRef::new("a-gpu"),
                id: "qwen3:8b".into(),
            }],
            ..Policy::default()
        };
        let d = select(&req, &c, &policy).unwrap();
        assert_eq!(d.model.endpoint, EndpointRef::new("b-gpu"));
        assert!(d.fallbacks.is_empty());
    }

    #[test]
    fn a_shared_id_ranks_deterministically_by_endpoint() {
        let req = for_task(TaskKind::Chat).build();
        let a = on(m("qwen3:8b", Tier::Medium), "a-gpu");
        let b = on(m("qwen3:8b", Tier::Medium), "b-gpu");
        let d1 = select(&req, &[a.clone(), b.clone()], &Policy::default()).unwrap();
        let d2 = select(&req, &[b, a], &Policy::default()).unwrap();
        assert_eq!(d1, d2);
        assert_eq!(d1.model.endpoint, EndpointRef::new("a-gpu"));
    }

    #[test]
    fn decisions_and_no_candidate_serialize_to_json() {
        let req = for_task(TaskKind::Chat).tools().build();
        let c = [with_unknown(
            m("compat", Tier::Medium),
            &[Capability::Tools],
        )];
        let d = select(&req, &c, &Policy::default()).unwrap();
        let v = serde_json::to_value(&d).unwrap();
        assert_eq!(v["model"]["id"], "compat");
        assert_eq!(v["model"]["availability"], "available");
        assert_eq!(v["reason"][0], serde_json::json!({ "required": "tools" }));
        let req = for_task(TaskKind::Chat).min_context(64).build();
        let mut small = m("small", Tier::Medium);
        small.context_window = Some(8);
        let err = select(&req, &[small], &Policy::default()).unwrap_err();
        let v = serde_json::to_value(&err).unwrap();
        assert_eq!(v["unmet"][0]["need"], serde_json::json!({ "context": 64 }));
    }

    #[test]
    fn unmet_needs_lists_only_definite_failures() {
        let req = for_task(TaskKind::Chat)
            .vision()
            .tools()
            .audio()
            .min_context(32_000)
            .build();
        let mut p = with_caps(m("p", Tier::Medium), &[Capability::Vision]);
        p.unknown_capabilities.insert(Capability::Tools);
        p.context_window = Some(8_192);
        assert_eq!(
            unmet_needs(&req, &p),
            vec![HardNeed::Audio, HardNeed::Context(32_000)]
        );
        p.context_window = None;
        assert_eq!(unmet_needs(&req, &p), vec![HardNeed::Audio]);
    }

    #[test]
    fn find_looks_up_by_key() {
        let c = [
            on(m("qwen3:8b", Tier::Medium), "a-gpu"),
            on(m("qwen3:8b", Tier::Large), "b-gpu"),
        ];
        let k = ModelKey {
            endpoint: EndpointRef::new("b-gpu"),
            id: "qwen3:8b".into(),
        };
        assert_eq!(find(&c, &k).map(|p| p.tier), Some(Tier::Large));
        assert!(find(&c, &key("missing")).is_none());
    }

    #[test]
    fn a_kept_sticky_model_does_not_claim_nothing_better_qualified() {
        let req = for_task(TaskKind::Chat).min_context(32_000).build();
        let mut big = m("big", Tier::Medium);
        big.context_window = Some(131_072);
        let policy = Policy {
            sticky_model: Some(key("current")),
            ..Policy::default()
        };
        let d = select(&req, &[big, m("current", Tier::Medium)], &policy).unwrap();
        assert_eq!(d.model.id, "current");
        assert_eq!(
            d.reason,
            vec![
                ReasonPart::Required(HardNeed::Context(32_000)),
                ReasonPart::Sticky
            ]
        );
    }

    #[test]
    fn a_sticky_key_not_among_candidates_falls_back_to_ranking() {
        let req = for_task(TaskKind::Chat).build();
        let policy = Policy {
            sticky_model: Some(key("gone")),
            ..Policy::default()
        };
        let d = select(&req, &[m("a", Tier::Medium), m("b", Tier::Small)], &policy).unwrap();
        assert_eq!(d.model.id, "a");
        assert!(!d.reason.contains(&ReasonPart::Sticky));
    }

    #[test]
    fn no_candidates_at_all_still_names_every_need() {
        let req = for_task(TaskKind::Chat).vision().tools().build();
        let err = select(&req, &[], &Policy::default()).unwrap_err();
        assert_eq!(err.considered, 0);
        assert_eq!(
            err.unmet,
            vec![
                Unmet {
                    need: HardNeed::Vision,
                    near_miss: None
                },
                Unmet {
                    need: HardNeed::Tools,
                    near_miss: None
                },
            ]
        );
        assert_eq!(
            err.to_string(),
            "no candidate models are available (needed: vision, tool calling)"
        );
    }
}
