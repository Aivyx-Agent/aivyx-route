//! The stateful layer both products wrap around [`select`]: per-session
//! stickiness, operator pins, failure cooldowns, and each session's last
//! decision. Synchronous and clock-injected — callers pass `now` — so it
//! stays I/O-free and deterministic under test.
//!
//! Semantics (operator-approved in model-routing Part 2):
//! - cooling models are a last resort, never a reason to fail;
//! - a fallback, or a choice made while any model was cooling, never
//!   becomes a session's sticky model;
//! - a pin never writes the sticky map, and `unpin` clears it too.

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::profile::{Availability, Locality, ModelKey, ModelProfile, Tier};
use crate::requirements::{Requirements, TaskKind, TaskOverrides};
use crate::residency::ResidencySnapshot;
use crate::select::{Policy, find, select, unmet_needs};
use crate::sessions::SessionMap;

/// How long a model that failed to answer is skipped.
pub const DEFAULT_COOLDOWN: Duration = Duration::from_secs(60);

/// What one call needs routed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteQuery {
    pub task: TaskKind,
    /// Stickiness key: calls sharing a session with a sticky task kind
    /// stay on one model. `None` for side calls.
    pub session: Option<String>,
    pub tools: bool,
    pub vision: bool,
    /// Becomes the minimum context window; `0` = no requirement.
    pub estimated_prompt_tokens: u32,
    /// A soft tier for this call (e.g. from the classifier); `None` = the task's default.
    pub tier: Option<Tier>,
}

impl RouteQuery {
    pub fn new(task: TaskKind) -> Self {
        RouteQuery {
            task,
            session: None,
            tools: false,
            vision: false,
            estimated_prompt_tokens: 0,
            tier: None,
        }
    }
}

/// What a session's most recent routed call went to, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RouteRecord {
    pub model: ModelKey,
    pub task: TaskKind,
    /// One human sentence: the decision's reason plus any fallback note.
    pub reason: String,
}

/// The ordered models to try for one call, and why. Pass it back to
/// [`Router::succeeded`] once a model has answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutePlan {
    pub chain: Vec<ModelKey>,
    pub reason: String,
    task: TaskKind,
    session: Option<String>,
    /// `Some` only for a sticky task kind with a session.
    sticky_session: Option<String>,
    /// Whether answering from `chain[0]` may move the session there.
    may_stick: bool,
}

/// No candidate meets the call's hard needs. The message names what is
/// missing; products append their own "how to fix" hint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoRoute(pub String);

impl fmt::Display for NoRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NoRoute {}

pub struct Router {
    tasks: TaskOverrides,
    cooldown: Duration,
    allow_cloud: bool,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    profiles: Vec<ModelProfile>,
    /// Session → the model its main thread is on. Per-session maps are
    /// capped ([`SessionMap`]): conversations end without telling us.
    sticky: SessionMap<ModelKey>,
    /// Session → an explicit operator pin.
    pins: SessionMap<ModelKey>,
    /// Model → when its cooldown (from a retryable failure) expires.
    cooling: HashMap<ModelKey, Instant>,
    last: SessionMap<RouteRecord>,
    /// The latest residency snapshot; empty until a product sets one.
    residency: ResidencySnapshot,
}

impl Router {
    pub fn new(profiles: Vec<ModelProfile>, tasks: TaskOverrides) -> Self {
        Router {
            tasks,
            cooldown: DEFAULT_COOLDOWN,
            allow_cloud: false,
            state: Mutex::new(State {
                profiles,
                ..State::default()
            }),
        }
    }

    pub fn with_cooldown(mut self, cooldown: Duration) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Keep per-session state (sticky model, pin, last decision) for at
    /// most `max` sessions each; past that, the least recently updated
    /// session is forgotten. Default [`crate::MAX_SESSIONS`].
    pub fn with_max_sessions(self, max: usize) -> Self {
        {
            let mut state = self.state.lock().unwrap();
            state.sticky = SessionMap::with_capacity(max);
            state.pins = SessionMap::with_capacity(max);
            state.last = SessionMap::with_capacity(max);
        }
        self
    }

    /// Whether cloud profiles may be chosen. Default `false`.
    pub fn with_allow_cloud(mut self, allow_cloud: bool) -> Self {
        self.allow_cloud = allow_cloud;
        self
    }

    pub fn profiles(&self) -> Vec<ModelProfile> {
        self.state.lock().unwrap().profiles.clone()
    }

    /// Replaces the candidates (e.g. after re-discovery) and clears every
    /// cooldown.
    pub fn set_profiles(&self, profiles: Vec<ModelProfile>) {
        let mut state = self.state.lock().unwrap();
        state.profiles = profiles;
        state.cooling.clear();
    }

    /// Replaces the residency snapshot `plan` scores with (products refresh
    /// it on a short TTL, never per call).
    pub fn set_residency(&self, snapshot: ResidencySnapshot) {
        self.state.lock().unwrap().residency = snapshot;
    }

    /// The current residency snapshot.
    pub fn residency(&self) -> ResidencySnapshot {
        self.state.lock().unwrap().residency.clone()
    }

    /// The model `session`'s main thread is on: its pin, else the model
    /// routing made it stick to.
    pub fn current(&self, session: &str) -> Option<ModelKey> {
        let state = self.state.lock().unwrap();
        state
            .pins
            .get(session)
            .or_else(|| state.sticky.get(session))
            .cloned()
    }

    pub fn last_decision(&self, session: &str) -> Option<RouteRecord> {
        self.state.lock().unwrap().last.get(session).cloned()
    }

    /// Pins `session`'s main thread (sticky task kinds) to `key`. A pin
    /// to a cloud model is ignored (but kept) while cloud routing is off.
    /// Any sticky model is dropped: it's unused while pinned, and would
    /// otherwise resurface if the pin were later evicted by the session cap.
    pub fn pin(&self, session: &str, key: ModelKey) {
        let mut state = self.state.lock().unwrap();
        state.sticky.remove(session);
        state.pins.insert(session.to_string(), key);
    }

    /// Clears `session`'s pin and sticky model, so its next main-thread
    /// call is chosen by ranking. The last decision stays.
    pub fn unpin(&self, session: &str) {
        let mut state = self.state.lock().unwrap();
        state.pins.remove(session);
        state.sticky.remove(session);
    }

    pub fn pinned(&self, session: &str) -> Option<ModelKey> {
        self.state.lock().unwrap().pins.get(session).cloned()
    }

    /// A new conversation: drop the sticky model and last decision, keep
    /// any pin (an explicit operator choice).
    pub fn forget_session(&self, session: &str) {
        let mut state = self.state.lock().unwrap();
        state.sticky.remove(session);
        state.last.remove(session);
    }

    /// Records a retryable failure: `key` is skipped until `now + cooldown`
    /// (unless nothing else qualifies).
    pub fn failed(&self, key: &ModelKey, now: Instant) {
        self.state
            .lock()
            .unwrap()
            .cooling
            .insert(key.clone(), now + self.cooldown);
    }

    pub fn plan(&self, query: &RouteQuery, now: Instant) -> Result<RoutePlan, NoRoute> {
        let mut builder = Requirements::builder().task(&query.task, &self.tasks);
        if let Some(tier) = query.tier {
            builder = builder.tier(tier);
        }
        if query.tools {
            builder = builder.tools();
        }
        if query.vision {
            builder = builder.vision();
        }
        if query.estimated_prompt_tokens > 0 {
            builder = builder.min_context(query.estimated_prompt_tokens);
        }
        let req = builder.build();
        let sticky_session = query.session.clone().filter(|_| query.task.is_sticky());
        let mut state = self.state.lock().unwrap();
        state.cooling.retain(|_, until| *until > now);

        // A pin to a cloud model while cloud routing is off is ignored (but
        // kept): selection runs as if unpinned, and the reason says so.
        let mut ignored_pin = None;
        if let Some(session) = &sticky_session
            && let Some(pin) = state.pins.get(session).cloned()
        {
            // A pin in use stays recent, so the session cap evicts idle
            // pins first.
            state.pins.insert(session.clone(), pin.clone());
            let profile = find(&state.profiles, &pin);
            if profile.is_some_and(|p| p.locality == Locality::Cloud) && !self.allow_cloud {
                ignored_pin = Some(format!(
                    "ignored pin to `{pin}`: it's a cloud model and cloud routing is off; "
                ));
            } else {
                return Ok(self.pinned_plan(query, &req, pin, profile, sticky_session));
            }
        }

        let any_cooling = state
            .profiles
            .iter()
            .any(|p| state.cooling.contains_key(&p.key()));
        let without_cooling: Vec<ModelProfile> = state
            .profiles
            .iter()
            .cloned()
            .map(|mut p| {
                if state.cooling.contains_key(&p.key()) {
                    p.availability = Availability::Unavailable;
                }
                p
            })
            .collect();
        let policy = Policy {
            sticky_model: sticky_session
                .as_ref()
                .and_then(|s| state.sticky.get(s).cloned()),
            allow_cloud: self.allow_cloud,
            exclude: Vec::new(),
            residency: state.residency.clone(),
        };
        let (decision, retrying_cooled) = match select(&req, &without_cooling, &policy) {
            Ok(decision) => (decision, false),
            Err(_) if any_cooling => (
                select(&req, &state.profiles, &policy).map_err(|e| NoRoute(e.to_string()))?,
                true,
            ),
            Err(e) => return Err(NoRoute(e.to_string())),
        };
        let mut chain = vec![decision.model.key()];
        chain.extend(decision.fallbacks.iter().map(ModelProfile::key));
        let mut reason = ignored_pin.unwrap_or_default();
        reason.push_str(&decision.to_string());
        if retrying_cooled {
            reason.push_str("; retrying a model still cooling down after a failure");
        }
        Ok(RoutePlan {
            chain,
            reason,
            task: query.task.clone(),
            session: query.session.clone(),
            sticky_session,
            may_stick: !any_cooling,
        })
    }

    /// The plan for a pin that applies. Pins win, but a mismatch is
    /// flagged rather than hidden.
    fn pinned_plan(
        &self,
        query: &RouteQuery,
        req: &Requirements,
        pin: ModelKey,
        profile: Option<&ModelProfile>,
        sticky_session: Option<String>,
    ) -> RoutePlan {
        let mut reason = format!("pinned to `{pin}`");
        match profile {
            Some(profile) => {
                let unmet: Vec<String> = unmet_needs(req, profile)
                    .iter()
                    .map(ToString::to_string)
                    .collect();
                if !unmet.is_empty() {
                    reason.push_str(&format!("; warning: it may lack {}", unmet.join(", ")));
                }
            }
            None => reason.push_str("; warning: it isn't among the current candidates"),
        }
        RoutePlan {
            chain: vec![pin],
            reason,
            task: query.task.clone(),
            session: query.session.clone(),
            sticky_session,
            may_stick: false,
        }
    }

    /// `served` answered the call planned by `plan`, after `failures`
    /// (human-readable, one per failed attempt). Updates stickiness and the
    /// session's last decision; returns the record.
    pub fn succeeded(
        &self,
        plan: &RoutePlan,
        served: &ModelKey,
        failures: &[String],
    ) -> RouteRecord {
        let mut reason = plan.reason.clone();
        if !failures.is_empty() {
            reason.push_str(&format!("; fell back after {} failed", failures.join(", ")));
        }
        let record = RouteRecord {
            model: served.clone(),
            task: plan.task.clone(),
            reason,
        };
        let mut state = self.state.lock().unwrap();
        if let Some(session) = &plan.sticky_session
            && plan.may_stick
            && *served == plan.chain[0]
        {
            state.sticky.insert(session.clone(), served.clone());
        }
        if let Some(session) = &plan.session {
            state.last.insert(session.clone(), record.clone());
        }
        record
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{Capability, EndpointRef, Tier};
    use crate::residency::{ModelResidency, ResidencySnapshot};

    fn key(endpoint: &str, id: &str) -> ModelKey {
        ModelKey {
            endpoint: EndpointRef::new(endpoint),
            id: id.into(),
        }
    }

    fn p(endpoint: &str, id: &str, tier: Tier, caps: &[Capability]) -> ModelProfile {
        let mut p = ModelProfile::new(id, EndpointRef::new(endpoint));
        p.tier = tier;
        p.capabilities.insert(Capability::Completion);
        p.capabilities.extend(caps.iter().copied());
        p
    }

    /// default@backend (Medium), big@gpu (Large), tiny@gpu (Small).
    fn three() -> Vec<ModelProfile> {
        vec![
            p("backend", "default", Tier::Medium, &[]),
            p("gpu", "big", Tier::Large, &[]),
            p("gpu", "tiny", Tier::Small, &[]),
        ]
    }

    fn router(profiles: Vec<ModelProfile>) -> Router {
        Router::new(profiles, TaskOverrides::default())
    }

    fn q(task: TaskKind, session: Option<&str>) -> RouteQuery {
        RouteQuery {
            session: session.map(str::to_string),
            ..RouteQuery::new(task)
        }
    }

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    #[test]
    fn per_session_state_is_capped_and_the_idlest_session_goes_first() {
        let r = router(three()).with_max_sessions(2);
        let t0 = Instant::now();
        for s in ["s1", "s2", "s3"] {
            let plan = r.plan(&q(TaskKind::CodeEdit, Some(s)), t0).unwrap();
            r.succeeded(&plan, &plan.chain[0].clone(), &[]);
        }
        assert_eq!(r.current("s1"), None, "the idlest session was evicted");
        assert!(r.last_decision("s1").is_none());
        assert_eq!(r.current("s3"), Some(key("gpu", "big")));
        assert!(r.last_decision("s2").is_some());
        // Pins are capped the same way.
        for s in ["p1", "p2", "p3"] {
            r.pin(s, key("gpu", "tiny"));
        }
        assert_eq!(r.pinned("p1"), None);
        assert_eq!(r.pinned("p3"), Some(key("gpu", "tiny")));
    }

    #[test]
    fn an_evicted_pin_never_resurfaces_an_older_sticky_model() {
        let r = router(three()).with_max_sessions(1);
        let t0 = Instant::now();
        let plan = r.plan(&q(TaskKind::CodeEdit, Some("s")), t0).unwrap();
        r.succeeded(&plan, &key("gpu", "big"), &[]);
        r.pin("s", key("gpu", "tiny"));
        r.pin("other", key("gpu", "tiny")); // evicts s's pin
        assert_eq!(r.pinned("s"), None);
        assert_eq!(r.current("s"), None, "no stale pre-pin sticky model");
    }

    #[test]
    fn a_pin_in_use_stays_recent() {
        let r = router(three()).with_max_sessions(2);
        let t0 = Instant::now();
        r.pin("busy", key("gpu", "tiny"));
        r.pin("idle", key("gpu", "tiny"));
        // Routing through busy's pin keeps it recent…
        r.plan(&q(TaskKind::CodeEdit, Some("busy")), t0).unwrap();
        r.pin("new", key("gpu", "tiny"));
        // …so the idle pin goes, not the one in use.
        assert_eq!(r.pinned("busy"), Some(key("gpu", "tiny")));
        assert_eq!(r.pinned("idle"), None);
    }

    #[test]
    fn the_main_thread_sticks_and_side_calls_route_freely() {
        let r = router(three());
        let t0 = Instant::now();
        let plan = r.plan(&q(TaskKind::CodeEdit, Some("s")), t0).unwrap();
        assert_eq!(plan.chain[0], key("gpu", "big"));
        r.succeeded(&plan, &key("gpu", "big"), &[]);
        assert_eq!(r.current("s"), Some(key("gpu", "big")));
        // Chat wants Medium but the session stays on big.
        let plan = r.plan(&q(TaskKind::Chat, Some("s")), t0).unwrap();
        assert_eq!(plan.chain[0], key("gpu", "big"));
        // A side call routes on its own merits and does not move the session.
        let plan = r.plan(&q(TaskKind::Summarize, Some("s")), t0).unwrap();
        assert_eq!(plan.chain[0], key("gpu", "tiny"));
        r.succeeded(&plan, &key("gpu", "tiny"), &[]);
        assert_eq!(r.current("s"), Some(key("gpu", "big")));
    }

    #[test]
    fn hard_needs_come_from_the_query() {
        let mut profiles = three();
        profiles.push(p("gpu", "caller", Tier::Small, &[Capability::Tools]));
        profiles.push(p("gpu", "seer", Tier::Small, &[Capability::Vision]));
        let mut long = p("gpu", "long", Tier::Small, &[]);
        long.context_window = Some(131_072);
        profiles.push(long);
        profiles[0].context_window = Some(8_192);
        let r = router(profiles);
        let now = Instant::now();
        let tools = RouteQuery {
            tools: true,
            ..RouteQuery::new(TaskKind::Chat)
        };
        assert_eq!(r.plan(&tools, now).unwrap().chain[0], key("gpu", "caller"));
        let vision = RouteQuery {
            vision: true,
            ..RouteQuery::new(TaskKind::Chat)
        };
        assert_eq!(r.plan(&vision, now).unwrap().chain[0], key("gpu", "seer"));
        let big_prompt = RouteQuery {
            estimated_prompt_tokens: 20_000,
            ..RouteQuery::new(TaskKind::Summarize)
        };
        assert_eq!(
            r.plan(&big_prompt, now).unwrap().chain[0],
            key("gpu", "long")
        );
    }

    #[test]
    fn no_route_names_the_unmet_need() {
        let r = router(three());
        let tools = RouteQuery {
            tools: true,
            ..RouteQuery::new(TaskKind::Chat)
        };
        let err = r.plan(&tools, Instant::now()).unwrap_err();
        assert!(err.to_string().contains("tool calling"), "{err}");
    }

    #[test]
    fn a_failed_model_is_skipped_until_its_cooldown_expires() {
        let r = router(three());
        let t0 = Instant::now();
        r.failed(&key("gpu", "big"), t0);
        let plan = r.plan(&q(TaskKind::CodeEdit, None), t0 + secs(1)).unwrap();
        assert_ne!(plan.chain[0], key("gpu", "big"));
        assert!(!plan.chain.contains(&key("gpu", "big")));
        let plan = r.plan(&q(TaskKind::CodeEdit, None), t0 + secs(61)).unwrap();
        assert_eq!(plan.chain[0], key("gpu", "big"));
    }

    #[test]
    fn cooling_models_are_a_last_resort_not_a_blocker() {
        let mut profiles = three();
        profiles[0].availability = crate::profile::Availability::Unavailable;
        profiles.remove(2);
        let r = router(profiles);
        let t0 = Instant::now();
        r.failed(&key("gpu", "big"), t0);
        let plan = r.plan(&q(TaskKind::CodeEdit, None), t0 + secs(1)).unwrap();
        assert_eq!(plan.chain, vec![key("gpu", "big")]);
        assert!(
            plan.reason.contains("retrying a model still cooling down"),
            "{}",
            plan.reason
        );
    }

    #[test]
    fn a_fallback_is_recorded_but_never_becomes_sticky() {
        let r = router(three());
        let t0 = Instant::now();
        let plan = r.plan(&q(TaskKind::CodeEdit, Some("s")), t0).unwrap();
        assert_eq!(plan.chain[0], key("gpu", "big"));
        r.failed(&key("gpu", "big"), t0);
        let record = r.succeeded(
            &plan,
            &key("backend", "default"),
            &["`big@gpu` (down)".to_string()],
        );
        assert_eq!(record.model, key("backend", "default"));
        assert!(
            record
                .reason
                .contains("fell back after `big@gpu` (down) failed"),
            "{}",
            record.reason
        );
        assert_eq!(r.current("s"), None);
        assert_eq!(r.last_decision("s"), Some(record));
    }

    #[test]
    fn a_choice_made_while_a_model_cools_does_not_move_the_session() {
        let r = router(three());
        let t0 = Instant::now();
        let plan = r.plan(&q(TaskKind::CodeEdit, Some("s")), t0).unwrap();
        r.succeeded(&plan, &key("gpu", "big"), &[]);
        r.failed(&key("gpu", "big"), t0);
        let plan = r
            .plan(&q(TaskKind::CodeEdit, Some("s")), t0 + secs(1))
            .unwrap();
        assert_ne!(plan.chain[0], key("gpu", "big"));
        let served = plan.chain[0].clone();
        r.succeeded(&plan, &served, &[]);
        assert_eq!(r.current("s"), Some(key("gpu", "big")));
        let plan = r
            .plan(&q(TaskKind::CodeEdit, Some("s")), t0 + secs(61))
            .unwrap();
        assert_eq!(plan.chain[0], key("gpu", "big"));
    }

    #[test]
    fn a_pin_overrides_selection_warns_and_unpin_returns_to_ranking() {
        let mut profiles = three();
        profiles[1].capabilities.insert(Capability::Tools);
        let r = router(profiles);
        let now = Instant::now();
        r.pin("s", key("gpu", "tiny"));
        let query = RouteQuery {
            tools: true,
            ..q(TaskKind::CodeEdit, Some("s"))
        };
        let plan = r.plan(&query, now).unwrap();
        assert_eq!(plan.chain, vec![key("gpu", "tiny")]);
        assert!(
            plan.reason.contains("pinned to `tiny@gpu`"),
            "{}",
            plan.reason
        );
        assert!(
            plan.reason.contains("may lack tool calling"),
            "{}",
            plan.reason
        );
        r.succeeded(&plan, &key("gpu", "tiny"), &[]);
        assert_eq!(r.current("s"), Some(key("gpu", "tiny")));
        r.unpin("s");
        assert_eq!(r.current("s"), None);
        assert_eq!(r.plan(&query, now).unwrap().chain[0], key("gpu", "big"));
    }

    #[test]
    fn a_pin_to_a_model_that_is_no_longer_a_candidate_is_kept_but_flagged() {
        // Pins win (never silently overridden), but a refresh that drops the
        // pinned model must not leave a clean-looking reason behind.
        let r = router(three());
        r.pin("s", key("gpu", "tiny"));
        r.set_profiles(vec![p("backend", "default", Tier::Medium, &[])]);
        let plan = r
            .plan(&q(TaskKind::CodeEdit, Some("s")), Instant::now())
            .unwrap();
        assert_eq!(plan.chain, vec![key("gpu", "tiny")]);
        assert!(
            plan.reason
                .contains("warning: it isn't among the current candidates"),
            "{}",
            plan.reason
        );
    }

    #[test]
    fn a_pin_to_a_cloud_model_falls_back_to_selection_while_cloud_routing_is_off() {
        let mut profiles = three();
        let mut cloud = p("anthropic", "claude", Tier::Large, &[]);
        cloud.locality = Locality::Cloud;
        profiles.push(cloud);
        let r = router(profiles.clone());
        r.pin("s", key("anthropic", "claude"));
        let plan = r
            .plan(&q(TaskKind::CodeEdit, Some("s")), Instant::now())
            .unwrap();
        assert!(
            !plan.chain.contains(&key("anthropic", "claude")),
            "{:?}",
            plan.chain
        );
        assert_eq!(plan.chain[0], key("gpu", "big"));
        assert!(
            plan.reason.starts_with(
                "ignored pin to `claude@anthropic`: it's a cloud model and cloud routing is off; chose"
            ),
            "{}",
            plan.reason
        );
        // The pin is kept for when cloud routing is on.
        assert_eq!(r.pinned("s"), Some(key("anthropic", "claude")));
        let allowed = router(profiles).with_allow_cloud(true);
        allowed.pin("s", key("anthropic", "claude"));
        let plan = allowed
            .plan(&q(TaskKind::CodeEdit, Some("s")), Instant::now())
            .unwrap();
        assert_eq!(plan.chain, vec![key("anthropic", "claude")]);
        assert!(!plan.reason.contains("warning"), "{}", plan.reason);
    }

    #[test]
    fn a_pin_does_not_capture_side_calls() {
        let r = router(three());
        r.pin("s", key("gpu", "big"));
        let plan = r
            .plan(&q(TaskKind::Summarize, Some("s")), Instant::now())
            .unwrap();
        assert_eq!(plan.chain[0], key("gpu", "tiny"));
    }

    #[test]
    fn forget_session_keeps_the_pin() {
        let r = router(three());
        let now = Instant::now();
        let plan = r.plan(&q(TaskKind::CodeEdit, Some("s")), now).unwrap();
        r.succeeded(&plan, &key("gpu", "big"), &[]);
        r.pin("s", key("gpu", "tiny"));
        r.forget_session("s");
        assert!(r.last_decision("s").is_none());
        assert_eq!(r.pinned("s"), Some(key("gpu", "tiny")));
    }

    #[test]
    fn cloud_models_need_allow_cloud() {
        let mut profiles = three();
        let mut cloud = p("default", "claude", Tier::Large, &[]);
        cloud.locality = Locality::Cloud;
        profiles.push(cloud);
        profiles.remove(1);
        let now = Instant::now();
        let plan = router(profiles.clone())
            .plan(&q(TaskKind::Plan, None), now)
            .unwrap();
        assert_ne!(plan.chain[0], key("default", "claude"));
        let plan = router(profiles)
            .with_allow_cloud(true)
            .plan(&q(TaskKind::Plan, None), now)
            .unwrap();
        assert_eq!(plan.chain[0], key("default", "claude"));
    }

    #[test]
    fn set_profiles_replaces_candidates_and_clears_cooldowns() {
        let r = router(three());
        let t0 = Instant::now();
        r.failed(&key("gpu", "big"), t0);
        r.set_profiles(three());
        let plan = r.plan(&q(TaskKind::CodeEdit, None), t0 + secs(1)).unwrap();
        assert_eq!(plan.chain[0], key("gpu", "big"));
    }

    #[test]
    fn succeeded_records_the_decision_per_session() {
        let r = router(three());
        let plan = r
            .plan(&q(TaskKind::CodeEdit, Some("main")), Instant::now())
            .unwrap();
        let record = r.succeeded(&plan, &key("gpu", "big"), &[]);
        assert_eq!(record.task, TaskKind::CodeEdit);
        assert!(
            record.reason.starts_with("chose `big`"),
            "{}",
            record.reason
        );
        assert_eq!(r.last_decision("main"), Some(record));
        assert!(r.last_decision("other").is_none());
    }

    #[test]
    fn plan_scores_with_the_latest_residency() {
        let r = router(vec![
            p("gpu", "a", Tier::Medium, &[]),
            p("gpu", "b", Tier::Medium, &[]),
        ]);
        let t0 = Instant::now();
        assert_eq!(
            r.plan(&q(TaskKind::Chat, None), t0).unwrap().chain[0],
            key("gpu", "a")
        );
        let snapshot = ResidencySnapshot {
            models: [(key("gpu", "b"), ModelResidency::Loaded { vram_bytes: None })].into(),
            ..ResidencySnapshot::default()
        };
        r.set_residency(snapshot.clone());
        assert_eq!(r.residency(), snapshot);
        let plan = r.plan(&q(TaskKind::Chat, None), t0).unwrap();
        assert_eq!(plan.chain[0], key("gpu", "b"));
        assert!(plan.reason.contains("already loaded"), "{}", plan.reason);
        // Replacing the snapshot replaces it; an empty one restores the old order.
        r.set_residency(ResidencySnapshot::default());
        assert_eq!(
            r.plan(&q(TaskKind::Chat, None), t0).unwrap().chain[0],
            key("gpu", "a")
        );
    }

    #[test]
    fn a_soft_tier_override_chooses_by_tier() {
        let r = router(three());
        let now = Instant::now();
        // Chat defaults to Medium, so it chooses default@backend.
        let plan = r.plan(&q(TaskKind::Chat, None), now).unwrap();
        assert_eq!(plan.chain[0], key("backend", "default"));
        // With tier: Some(Large), it chooses big@gpu.
        let query = RouteQuery {
            tier: Some(Tier::Large),
            ..q(TaskKind::Chat, None)
        };
        let plan = r.plan(&query, now).unwrap();
        assert_eq!(plan.chain[0], key("gpu", "big"));
    }

    #[test]
    fn route_query_new_has_no_tier() {
        let query = RouteQuery::new(TaskKind::Chat);
        assert_eq!(query.tier, None);
    }
}
