//! Combine what discovery found with what the operator declared.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::config::{DefaultEndpoint, EndpointKind, RosterEntry, RoutingConfig};
use crate::profile::{
    Availability, Capability, CapabilitySet, EndpointRef, Locality, ModelProfile,
};

/// What one backend said about one model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiscoveredModel {
    pub id: String,
    pub capabilities: CapabilitySet,
    /// Capabilities this source cannot report on.
    pub unknown_capabilities: CapabilitySet,
    pub context_window: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryOutcome {
    Reached(Vec<DiscoveredModel>),
    /// The endpoint could not be queried; the string says why.
    Unreachable(String),
    /// Deliberately not probed (cloud endpoints).
    NotProbed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiscoveryReport {
    pub endpoint: EndpointRef,
    pub outcome: DiscoveryOutcome,
}

/// Build the candidate list: every discovered model, enriched/overridden by
/// roster entries, plus roster-only models. Sorted by `(endpoint, id)`.
///
/// An endpoint that is neither `default` nor a `[routing.endpoints]` key
/// fails closed: its models are `Cloud` and `Unavailable`. An explicit
/// roster `locality` still overrides the locality, never the availability.
///
/// Duplicate roster entries for the same (endpoint, id) apply in order, each
/// overriding only the fields it sets; a capability denied by any of them is
/// always removed.
pub fn merge(
    config: &RoutingConfig,
    default: &DefaultEndpoint,
    reports: &[DiscoveryReport],
) -> Vec<ModelProfile> {
    // `None` = an endpoint neither the default nor configured.
    let endpoint_kind = |ep: &EndpointRef| {
        if *ep == default.name {
            Some(default.kind)
        } else {
            config.endpoints.get(ep.as_str()).map(|c| c.kind)
        }
    };
    // Configured endpoints: their address decides (see
    // `EndpointConfig::effective_locality`). Unknown endpoints fail closed:
    // a typo must never make a cloud model look local.
    let endpoint_locality = |ep: &EndpointRef| {
        if *ep == default.name {
            default.kind.locality()
        } else {
            config
                .endpoints
                .get(ep.as_str())
                .map_or(Locality::Cloud, |c| c.effective_locality())
        }
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
                p.unknown_capabilities = d.unknown_capabilities.clone();
                p.context_window = d.context_window;
                p.source.discovered = true;
                profiles.insert((report.endpoint.clone(), d.id.clone()), p);
            }
        }
    }

    let roster_ep = |entry: &RosterEntry| {
        entry
            .endpoint
            .as_deref()
            .map(EndpointRef::new)
            .unwrap_or_else(|| default.name.clone())
    };

    // Resolve bare Ollama names against discovered `name:latest` once, up
    // front, so the outcome can't depend on roster order.
    let roster_keys: BTreeSet<(EndpointRef, String)> = config
        .models
        .iter()
        .map(|e| (roster_ep(e), e.id.clone()))
        .collect();
    let mut resolved: BTreeMap<(EndpointRef, String), String> = BTreeMap::new();
    for key in &roster_keys {
        if let Some(id) =
            resolve_ollama_latest(&mut profiles, key, &roster_keys, endpoint_kind(&key.0))
        {
            resolved.insert(key.clone(), id);
        }
    }

    // Collect all capability denies for each (endpoint, id) key.
    let mut all_denies: BTreeMap<(EndpointRef, String), CapabilitySet> = BTreeMap::new();

    for entry in &config.models {
        let ep = roster_ep(entry);
        let key = (ep.clone(), entry.id.clone());
        let key = match resolved.get(&key) {
            Some(id) => (ep.clone(), id.clone()),
            None => key,
        };

        // Track all denies for this key.
        all_denies
            .entry(key.clone())
            .or_default()
            .extend(entry.capabilities_deny.iter().copied());

        let p = profiles.entry(key).or_insert_with(|| {
            let mut p = ModelProfile::new(entry.id.clone(), ep.clone());
            p.locality = endpoint_locality(&ep);
            p.unknown_capabilities = Capability::ALL.into_iter().collect();
            p.availability = match outcome_of(&ep) {
                Some(DiscoveryOutcome::Unreachable(_)) => Availability::Unavailable,
                _ => Availability::Unverified,
            };
            p
        });
        p.capabilities.extend(entry.capabilities.iter().copied());
        p.unknown_capabilities
            .retain(|c| !entry.capabilities.contains(c) && !entry.capabilities_deny.contains(c));
        if entry.context_window.is_some() {
            p.context_window = entry.context_window;
        }
        if let Some(tier) = entry.tier {
            p.tier = tier;
        }
        if let Some(strengths) = &entry.strengths {
            p.strengths = strengths.clone();
        }
        if let Some(priority) = entry.priority {
            p.priority = priority;
        }
        if let Some(locality) = entry.locality {
            p.locality = locality;
        }
        p.source.in_roster = true;
    }

    // Apply all accumulated denies.
    for (key, denies) in all_denies {
        if let Some(p) = profiles.get_mut(&key) {
            p.capabilities.retain(|c| !denies.contains(c));
        }
    }

    for p in profiles.values_mut() {
        if endpoint_kind(&p.endpoint).is_none() {
            p.availability = Availability::Unavailable;
        }
    }

    profiles.into_values().collect()
}

/// Ollama reports a tagless model as `name:latest`, but operators write the
/// bare `name` (what `ollama run` and the products' `model` settings
/// accept). For a roster entry on an Ollama endpoint naming a bare `name`
/// that discovery didn't report, but whose `name:latest` it did:
///
/// - if the roster names only the bare `name`, the discovered profile is
///   re-keyed to it, so the entry enriches it under the operator's name;
/// - if the roster also names `name:latest`, both entries mean one model:
///   returns `name:latest`, the id the bare entry resolves to, so both
///   apply (in roster order) to the one discovered profile.
///
/// Anything else is left as written.
fn resolve_ollama_latest(
    profiles: &mut BTreeMap<(EndpointRef, String), ModelProfile>,
    key: &(EndpointRef, String),
    roster_keys: &BTreeSet<(EndpointRef, String)>,
    kind: Option<EndpointKind>,
) -> Option<String> {
    let (ep, id) = key;
    if kind != Some(EndpointKind::Ollama) || id.contains(':') || profiles.contains_key(key) {
        return None;
    }
    let latest = (ep.clone(), format!("{id}:latest"));
    if roster_keys.contains(&latest) {
        return profiles.contains_key(&latest).then(|| latest.1.clone());
    }
    let mut p = profiles.remove(&latest)?;
    p.id = id.clone();
    profiles.insert(key.clone(), p);
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DefaultEndpoint, EndpointConfig, EndpointKind, RosterEntry};
    use crate::profile::{ProfileSource, Strength, Tier};
    use std::collections::BTreeSet;

    fn ep(name: &str) -> EndpointRef {
        EndpointRef::new(name)
    }

    fn local_default(name: &str) -> DefaultEndpoint {
        DefaultEndpoint {
            name: ep(name),
            kind: EndpointKind::Ollama,
        }
    }

    fn found(id: &str, caps: &[Capability], ctx: Option<u32>) -> DiscoveredModel {
        DiscoveredModel {
            id: id.into(),
            capabilities: caps.iter().copied().collect(),
            unknown_capabilities: CapabilitySet::new(),
            context_window: ctx,
        }
    }

    /// An OpenAI-compat style discovery: nothing known.
    fn found_blind(id: &str) -> DiscoveredModel {
        DiscoveredModel {
            id: id.into(),
            capabilities: CapabilitySet::new(),
            unknown_capabilities: Capability::ALL.into_iter().collect(),
            context_window: None,
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
            strengths: None,
            priority: None,
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

    /// Ollama reports a tagless model as `name:latest`; a roster entry
    /// naming the bare `name` enriches that discovered profile (under the
    /// operator's name) instead of adding a second, blind one.
    #[test]
    fn a_bare_roster_name_adopts_the_discovered_ollama_latest_profile() {
        let reports = [reached(
            "main",
            vec![found(
                "llama3.2:latest",
                &[Capability::Tools],
                Some(131_072),
            )],
        )];
        let mut e = entry("llama3.2");
        e.tier = Some(Tier::Small);
        let ps = merge(&config_with(vec![e]), &local_default("main"), &reports);
        assert_eq!(ps.len(), 1, "got {ps:?}");
        let p = &ps[0];
        assert_eq!(p.id, "llama3.2");
        assert_eq!(p.capabilities, CapabilitySet::from([Capability::Tools]));
        assert_eq!(p.context_window, Some(131_072));
        assert_eq!(p.tier, Tier::Small);
        assert_eq!(p.availability, Availability::Available);
        assert_eq!(
            p.source,
            ProfileSource {
                discovered: true,
                in_roster: true
            }
        );
    }

    /// Listing both `m` and `m:latest` names one model: whatever the roster
    /// order, there is one profile, both entries apply, and every deny holds.
    #[test]
    fn bare_and_latest_entries_for_one_model_merge_in_either_order() {
        let reports = [reached(
            "main",
            vec![found(
                "m:latest",
                &[Capability::Tools, Capability::Vision],
                None,
            )],
        )];
        let mut bare = entry("m");
        bare.tier = Some(Tier::Large);
        let mut tagged = entry("m:latest");
        tagged.capabilities_deny = CapabilitySet::from([Capability::Vision]);
        for roster in [
            vec![bare.clone(), tagged.clone()],
            vec![tagged.clone(), bare.clone()],
        ] {
            let ps = merge(&config_with(roster), &local_default("main"), &reports);
            assert_eq!(ps.len(), 1, "got {ps:?}");
            let p = &ps[0];
            assert_eq!(p.id, "m:latest");
            assert_eq!(p.tier, Tier::Large);
            assert_eq!(p.capabilities, CapabilitySet::from([Capability::Tools]));
            assert_eq!(p.availability, Availability::Available);
        }
    }

    /// If discovery reported the bare name itself, nothing is adopted.
    #[test]
    fn a_discovered_bare_name_is_joined_exactly() {
        let reports = [reached(
            "main",
            vec![
                found("m", &[Capability::Tools], None),
                found("m:latest", &[], None),
            ],
        )];
        let ps = merge(
            &config_with(vec![entry("m")]),
            &local_default("main"),
            &reports,
        );
        let ids: Vec<&str> = ps.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, ["m", "m:latest"]);
        assert_eq!(
            find(&ps, "m").capabilities,
            CapabilitySet::from([Capability::Tools])
        );
    }

    #[test]
    fn latest_adoption_is_ollama_only_and_never_overrides_an_explicit_tag() {
        // A non-Ollama endpoint keeps both profiles.
        let default = DefaultEndpoint {
            name: ep("main"),
            kind: EndpointKind::OpenaiCompat,
        };
        let reports = [reached("main", vec![found("m:latest", &[], None)])];
        let ps = merge(&config_with(vec![entry("m")]), &default, &reports);
        assert_eq!(ps.len(), 2, "got {ps:?}");
        // An explicit `:latest` entry joins exactly; a different tag is not
        // adopted.
        let reports = [reached(
            "main",
            vec![found("m:latest", &[], None), found("q:8b", &[], None)],
        )];
        let ps = merge(
            &config_with(vec![entry("m:latest"), entry("q")]),
            &local_default("main"),
            &reports,
        );
        let ids: Vec<&str> = ps.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, ["m:latest", "q", "q:8b"]);
    }

    #[test]
    fn discovered_only_models_get_neutral_defaults() {
        let reports = [reached(
            "main",
            vec![found("qwen3:8b", &[Capability::Tools], Some(40_960))],
        )];
        let ps = merge(&RoutingConfig::default(), &local_default("main"), &reports);
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
        e.strengths = Some(BTreeSet::from([Strength::Chat]));
        e.priority = Some(3);
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
        let ps = merge(&config_with(vec![e]), &local_default("main"), &reports);
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
        let ps = merge(&config_with(vec![e]), &local_default("main"), &reports);
        assert_eq!(find(&ps, "m").context_window, Some(32_768));
    }

    #[test]
    fn roster_only_models_are_unverified() {
        let ps = merge(
            &config_with(vec![entry("ghost")]),
            &local_default("main"),
            &[],
        );
        assert_eq!(find(&ps, "ghost").availability, Availability::Unverified);
        assert_eq!(find(&ps, "ghost").endpoint, ep("main"));

        let reports = [reached("main", vec![])];
        let ps = merge(
            &config_with(vec![entry("ghost")]),
            &local_default("main"),
            &reports,
        );
        assert_eq!(find(&ps, "ghost").availability, Availability::Unverified);
    }

    #[test]
    fn roster_models_on_unreachable_endpoints_are_unavailable() {
        let reports = [DiscoveryReport {
            endpoint: ep("main"),
            outcome: DiscoveryOutcome::Unreachable("connection refused".into()),
        }];
        let ps = merge(
            &config_with(vec![entry("m")]),
            &local_default("main"),
            &reports,
        );
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
                locality: None,
            },
        );
        let mut claude = entry("claude-sonnet-5");
        claude.endpoint = Some("anthropic".into());
        let mut forced = entry("forced");
        forced.locality = Some(Locality::Cloud);
        config.models = vec![claude, forced, entry("local-one")];
        let ps = merge(&config, &local_default("main"), &[]);
        assert_eq!(find(&ps, "claude-sonnet-5").locality, Locality::Cloud);
        assert_eq!(find(&ps, "forced").locality, Locality::Cloud);
        assert_eq!(find(&ps, "local-one").locality, Locality::Local);
    }

    #[test]
    fn locality_follows_the_endpoint_address_not_just_its_kind() {
        use crate::requirements::{Requirements, TaskKind, TaskOverrides};
        use crate::select::{Policy, select};
        let config: RoutingConfig = toml::from_str(
            r#"
[endpoints.groq]
kind = "openai_compat"
base_url = "https://api.groq.com/openai/v1"

[endpoints.lan]
kind = "ollama"
base_url = "http://gpu.example.com:11434"
locality = "local"

[endpoints.home]
kind = "llama_router"
base_url = "http://192.168.1.20:8080"
"#,
        )
        .unwrap();
        let reports = [
            reached("groq", vec![found("llama-3.3-70b", &[], None)]),
            reached("lan", vec![found("qwen3:8b", &[], None)]),
            reached("home", vec![found("gemma", &[], None)]),
        ];
        let mut roster_only = entry("kimi");
        roster_only.endpoint = Some("groq".into());
        let config = RoutingConfig {
            models: vec![roster_only],
            ..config
        };
        let ps = merge(&config, &local_default("main"), &reports);
        assert_eq!(find(&ps, "llama-3.3-70b").locality, Locality::Cloud);
        assert_eq!(find(&ps, "kimi").locality, Locality::Cloud);
        assert_eq!(find(&ps, "qwen3:8b").locality, Locality::Local);
        assert_eq!(find(&ps, "gemma").locality, Locality::Local);

        let groq_only = merge(&config, &local_default("main"), &reports[..1]);
        let req = Requirements::builder()
            .task(&TaskKind::Chat, &TaskOverrides::default())
            .build();
        assert!(
            select(&req, &groq_only, &Policy::default()).is_err(),
            "cloud off must never pick a hosted endpoint"
        );
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
        let ps = merge(&RoutingConfig::default(), &local_default("main"), &reports);
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

    #[test]
    fn a_deny_from_any_duplicate_entry_wins() {
        let mut e1 = entry("m");
        e1.capabilities_deny = CapabilitySet::from([Capability::Tools]);

        let mut e2 = entry("m");
        e2.capabilities = CapabilitySet::from([Capability::Tools, Capability::Vision]);

        let reports = [reached(
            "main",
            vec![found(
                "m",
                &[Capability::Completion, Capability::Tools],
                None,
            )],
        )];

        // Test with entry1 first, entry2 second
        let ps = merge(
            &config_with(vec![e1.clone(), e2.clone()]),
            &local_default("main"),
            &reports,
        );
        let p = find(&ps, "m");
        assert_eq!(
            p.capabilities,
            CapabilitySet::from([Capability::Completion, Capability::Vision])
        );

        // Test with entry2 first, entry1 second (should get same result)
        let ps = merge(&config_with(vec![e2, e1]), &local_default("main"), &reports);
        let p = find(&ps, "m");
        assert_eq!(
            p.capabilities,
            CapabilitySet::from([Capability::Completion, Capability::Vision])
        );
    }

    #[test]
    fn duplicate_entries_apply_other_fields_in_order() {
        let mut e1 = entry("m");
        e1.tier = Some(Tier::Small);
        e1.priority = Some(1);

        let mut e2 = entry("m");
        e2.tier = Some(Tier::Large);
        e2.priority = Some(7);

        let reports = [reached("main", vec![found("m", &[], None)])];
        let ps = merge(&config_with(vec![e1, e2]), &local_default("main"), &reports);
        assert_eq!(ps.len(), 1);
        let p = find(&ps, "m");
        assert_eq!(p.tier, Tier::Large);
        assert_eq!(p.priority, 7);
    }

    #[test]
    fn not_probed_endpoints_leave_roster_models_unverified() {
        let reports = [DiscoveryReport {
            endpoint: ep("main"),
            outcome: DiscoveryOutcome::NotProbed,
        }];
        let ps = merge(
            &config_with(vec![entry("m")]),
            &local_default("main"),
            &reports,
        );
        assert_eq!(find(&ps, "m").availability, Availability::Unverified);
    }

    #[test]
    fn discovered_unknown_capabilities_are_carried_over() {
        let reports = [reached("main", vec![found_blind("compat")])];
        let ps = merge(&RoutingConfig::default(), &local_default("main"), &reports);
        assert_eq!(
            find(&ps, "compat").unknown_capabilities,
            Capability::ALL.into_iter().collect::<CapabilitySet>()
        );
    }

    #[test]
    fn roster_only_models_start_with_every_capability_unknown() {
        let ps = merge(
            &config_with(vec![entry("ghost")]),
            &local_default("main"),
            &[],
        );
        assert_eq!(
            find(&ps, "ghost").unknown_capabilities,
            Capability::ALL.into_iter().collect::<CapabilitySet>()
        );
    }

    #[test]
    fn roster_declarations_and_denies_make_capabilities_known() {
        let mut e = entry("compat");
        e.capabilities = CapabilitySet::from([Capability::Tools]);
        e.capabilities_deny = CapabilitySet::from([Capability::Vision]);
        let reports = [reached("main", vec![found_blind("compat")])];
        let ps = merge(&config_with(vec![e]), &local_default("main"), &reports);
        let p = find(&ps, "compat");
        assert_eq!(p.capabilities, CapabilitySet::from([Capability::Tools]));
        assert_eq!(
            p.unknown_capabilities,
            CapabilitySet::from([
                Capability::Completion,
                Capability::Thinking,
                Capability::Audio,
                Capability::Embedding
            ])
        );
    }

    #[test]
    fn denying_an_unknown_capability_makes_a_request_needing_it_fail() {
        use crate::requirements::{Requirements, TaskKind, TaskOverrides};
        use crate::select::{Policy, select};
        let mut e = entry("compat");
        e.capabilities_deny = CapabilitySet::from([Capability::Tools]);
        let reports = [reached("main", vec![found_blind("compat")])];
        let ps = merge(&config_with(vec![e]), &local_default("main"), &reports);
        let req = Requirements::builder()
            .task(&TaskKind::Chat, &TaskOverrides::default())
            .tools()
            .build();
        assert!(select(&req, &ps, &Policy::default()).is_err());
        let ps = merge(&RoutingConfig::default(), &local_default("main"), &reports);
        assert!(select(&req, &ps, &Policy::default()).is_ok());
    }

    #[test]
    fn an_unknown_endpoint_fails_closed() {
        let mut typo = entry("m");
        typo.endpoint = Some("olama-main".into());
        let ps = merge(&config_with(vec![typo]), &local_default("main"), &[]);
        let p = find(&ps, "m");
        assert_eq!(p.locality, Locality::Cloud);
        assert_eq!(p.availability, Availability::Unavailable);

        use crate::requirements::{Requirements, TaskKind, TaskOverrides};
        use crate::select::{Policy, select};
        let req = Requirements::builder()
            .task(&TaskKind::Chat, &TaskOverrides::default())
            .build();
        let allow = Policy {
            allow_cloud: true,
            ..Policy::default()
        };
        assert!(select(&req, &ps, &allow).is_err());
    }

    #[test]
    fn an_explicit_locality_does_not_make_an_unknown_endpoint_available() {
        let mut typo = entry("m");
        typo.endpoint = Some("olama-main".into());
        typo.locality = Some(Locality::Local);
        let ps = merge(&config_with(vec![typo]), &local_default("main"), &[]);
        let p = find(&ps, "m");
        assert_eq!(p.locality, Locality::Local);
        assert_eq!(p.availability, Availability::Unavailable);
    }

    #[test]
    fn a_cloud_default_endpoint_makes_its_models_cloud() {
        let default = DefaultEndpoint {
            name: ep("anthropic"),
            kind: EndpointKind::Anthropic,
        };
        let ps = merge(&config_with(vec![entry("claude-sonnet-5")]), &default, &[]);
        let p = find(&ps, "claude-sonnet-5");
        assert_eq!(p.locality, Locality::Cloud);
        assert_eq!(p.availability, Availability::Unverified);
    }

    #[test]
    fn a_later_duplicate_entry_overrides_only_the_fields_it_sets() {
        let mut e1 = entry("m");
        e1.strengths = Some(BTreeSet::from([Strength::Code]));
        e1.priority = Some(4);
        e1.tier = Some(Tier::Small);
        let mut e2 = entry("m");
        e2.tier = Some(Tier::Large);
        let reports = [reached("main", vec![found("m", &[], None)])];
        let ps = merge(&config_with(vec![e1, e2]), &local_default("main"), &reports);
        let p = find(&ps, "m");
        assert_eq!(p.tier, Tier::Large);
        assert_eq!(p.strengths, BTreeSet::from([Strength::Code]));
        assert_eq!(p.priority, 4);
    }
}
