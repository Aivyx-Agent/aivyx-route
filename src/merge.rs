//! Combine what discovery found with what the operator declared.

use std::collections::BTreeMap;

use crate::config::RoutingConfig;
use crate::profile::{Availability, CapabilitySet, EndpointRef, Locality, ModelProfile};

/// What one backend said about one model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredModel {
    pub id: String,
    pub capabilities: CapabilitySet,
    pub context_window: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryOutcome {
    Reached(Vec<DiscoveredModel>),
    /// The endpoint could not be queried; the string says why.
    Unreachable(String),
    /// Deliberately not probed (cloud endpoints).
    NotProbed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveryReport {
    pub endpoint: EndpointRef,
    pub outcome: DiscoveryOutcome,
}

/// Build the candidate list: every discovered model, enriched/overridden by
/// roster entries, plus roster-only models. Sorted by `(endpoint, id)`.
pub fn merge(
    config: &RoutingConfig,
    default_endpoint: &EndpointRef,
    reports: &[DiscoveryReport],
) -> Vec<ModelProfile> {
    let endpoint_locality = |ep: &EndpointRef| {
        config
            .endpoints
            .get(ep.as_str())
            .map_or(Locality::Local, |c| c.kind.locality())
    };
    let outcome_of = |ep: &EndpointRef| {
        reports
            .iter()
            .find(|r| &r.endpoint == ep)
            .map(|r| &r.outcome)
    };

    let mut profiles: BTreeMap<(EndpointRef, String), ModelProfile> = BTreeMap::new();

    for report in reports {
        if let DiscoveryOutcome::Reached(models) = &report.outcome {
            for d in models {
                let mut p = ModelProfile::new(d.id.clone(), report.endpoint.clone());
                p.locality = endpoint_locality(&report.endpoint);
                p.capabilities = d.capabilities.clone();
                p.context_window = d.context_window;
                p.source.discovered = true;
                profiles.insert((report.endpoint.clone(), d.id.clone()), p);
            }
        }
    }

    for entry in &config.models {
        let ep = entry
            .endpoint
            .as_deref()
            .map(EndpointRef::new)
            .unwrap_or_else(|| default_endpoint.clone());
        let p = profiles
            .entry((ep.clone(), entry.id.clone()))
            .or_insert_with(|| {
                let mut p = ModelProfile::new(entry.id.clone(), ep.clone());
                p.locality = endpoint_locality(&ep);
                p.availability = match outcome_of(&ep) {
                    Some(DiscoveryOutcome::Unreachable(_)) => Availability::Unavailable,
                    _ => Availability::Unverified,
                };
                p
            });
        p.capabilities.extend(entry.capabilities.iter().copied());
        p.capabilities
            .retain(|c| !entry.capabilities_deny.contains(c));
        if entry.context_window.is_some() {
            p.context_window = entry.context_window;
        }
        if let Some(tier) = entry.tier {
            p.tier = tier;
        }
        p.strengths = entry.strengths.clone();
        p.priority = entry.priority;
        if let Some(locality) = entry.locality {
            p.locality = locality;
        }
        p.source.in_roster = true;
    }

    profiles.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EndpointConfig, EndpointKind, RosterEntry};
    use crate::profile::{Capability, ProfileSource, Strength, Tier};
    use std::collections::BTreeSet;

    fn ep(name: &str) -> EndpointRef {
        EndpointRef::new(name)
    }

    fn found(id: &str, caps: &[Capability], ctx: Option<u32>) -> DiscoveredModel {
        DiscoveredModel {
            id: id.into(),
            capabilities: caps.iter().copied().collect(),
            context_window: ctx,
        }
    }

    fn reached(endpoint: &str, models: Vec<DiscoveredModel>) -> DiscoveryReport {
        DiscoveryReport {
            endpoint: ep(endpoint),
            outcome: DiscoveryOutcome::Reached(models),
        }
    }

    fn entry(id: &str) -> RosterEntry {
        RosterEntry {
            id: id.into(),
            endpoint: None,
            locality: None,
            tier: None,
            strengths: BTreeSet::new(),
            priority: 0,
            capabilities: CapabilitySet::new(),
            capabilities_deny: CapabilitySet::new(),
            context_window: None,
        }
    }

    fn config_with(models: Vec<RosterEntry>) -> RoutingConfig {
        RoutingConfig {
            models,
            ..RoutingConfig::default()
        }
    }

    fn find<'a>(ps: &'a [ModelProfile], id: &str) -> &'a ModelProfile {
        ps.iter().find(|p| p.id == id).unwrap()
    }

    #[test]
    fn discovered_only_models_get_neutral_defaults() {
        let reports = [reached(
            "main",
            vec![found("qwen3:8b", &[Capability::Tools], Some(40_960))],
        )];
        let ps = merge(&RoutingConfig::default(), &ep("main"), &reports);
        let p = find(&ps, "qwen3:8b");
        assert_eq!(p.capabilities, CapabilitySet::from([Capability::Tools]));
        assert_eq!(p.context_window, Some(40_960));
        assert_eq!(p.tier, Tier::Medium);
        assert!(p.strengths.is_empty());
        assert_eq!(p.priority, 0);
        assert_eq!(p.availability, Availability::Available);
        assert_eq!(
            p.source,
            ProfileSource {
                discovered: true,
                in_roster: false
            }
        );
    }

    #[test]
    fn roster_adds_and_denies_capabilities_and_sets_judgements() {
        let mut e = entry("llava:13b");
        e.tier = Some(Tier::Small);
        e.strengths = BTreeSet::from([Strength::Chat]);
        e.priority = 3;
        e.capabilities = CapabilitySet::from([Capability::Thinking]);
        e.capabilities_deny = CapabilitySet::from([Capability::Tools]);
        let reports = [reached(
            "main",
            vec![found(
                "llava:13b",
                &[
                    Capability::Completion,
                    Capability::Vision,
                    Capability::Tools,
                ],
                Some(4_096),
            )],
        )];
        let ps = merge(&config_with(vec![e]), &ep("main"), &reports);
        let p = find(&ps, "llava:13b");
        assert_eq!(
            p.capabilities,
            CapabilitySet::from([
                Capability::Completion,
                Capability::Vision,
                Capability::Thinking
            ])
        );
        assert_eq!(p.tier, Tier::Small);
        assert_eq!(p.strengths, BTreeSet::from([Strength::Chat]));
        assert_eq!(p.priority, 3);
        assert_eq!(
            p.source,
            ProfileSource {
                discovered: true,
                in_roster: true
            }
        );
        assert_eq!(
            ps.len(),
            1,
            "roster entry must merge into the discovered profile"
        );
    }

    #[test]
    fn roster_context_window_overrides_discovery() {
        let mut e = entry("m");
        e.context_window = Some(32_768);
        let reports = [reached("main", vec![found("m", &[], Some(131_072))])];
        let ps = merge(&config_with(vec![e]), &ep("main"), &reports);
        assert_eq!(find(&ps, "m").context_window, Some(32_768));
    }

    #[test]
    fn roster_only_models_are_unverified() {
        let ps = merge(&config_with(vec![entry("ghost")]), &ep("main"), &[]);
        assert_eq!(find(&ps, "ghost").availability, Availability::Unverified);
        assert_eq!(find(&ps, "ghost").endpoint, ep("main"));

        let reports = [reached("main", vec![])];
        let ps = merge(&config_with(vec![entry("ghost")]), &ep("main"), &reports);
        assert_eq!(find(&ps, "ghost").availability, Availability::Unverified);
    }

    #[test]
    fn roster_models_on_unreachable_endpoints_are_unavailable() {
        let reports = [DiscoveryReport {
            endpoint: ep("main"),
            outcome: DiscoveryOutcome::Unreachable("connection refused".into()),
        }];
        let ps = merge(&config_with(vec![entry("m")]), &ep("main"), &reports);
        assert_eq!(find(&ps, "m").availability, Availability::Unavailable);
    }

    #[test]
    fn locality_comes_from_roster_then_endpoint_kind() {
        let mut config = RoutingConfig::default();
        config.endpoints.insert(
            "anthropic".into(),
            EndpointConfig {
                kind: EndpointKind::Anthropic,
                base_url: None,
            },
        );
        let mut claude = entry("claude-sonnet-5");
        claude.endpoint = Some("anthropic".into());
        let mut forced = entry("forced");
        forced.locality = Some(Locality::Cloud);
        config.models = vec![claude, forced, entry("local-one")];
        let ps = merge(&config, &ep("main"), &[]);
        assert_eq!(find(&ps, "claude-sonnet-5").locality, Locality::Cloud);
        assert_eq!(find(&ps, "forced").locality, Locality::Cloud);
        assert_eq!(find(&ps, "local-one").locality, Locality::Local);
    }

    #[test]
    fn same_id_on_different_endpoints_stays_separate_and_output_is_sorted() {
        let reports = [
            reached("b-gpu", vec![found("qwen3:8b", &[], None)]),
            reached(
                "a-gpu",
                vec![found("qwen3:8b", &[], None), found("aaa", &[], None)],
            ),
        ];
        let ps = merge(&RoutingConfig::default(), &ep("main"), &reports);
        let keys: Vec<(&str, &str)> = ps
            .iter()
            .map(|p| (p.endpoint.as_str(), p.id.as_str()))
            .collect();
        assert_eq!(
            keys,
            vec![
                ("a-gpu", "aaa"),
                ("a-gpu", "qwen3:8b"),
                ("b-gpu", "qwen3:8b")
            ]
        );
    }
}
