//! Task-aware model routing shared by `aivyx-pa` and `aivyx-coder`.
//!
//! Products describe candidate models as [`ModelProfile`]s, derive what a
//! request needs, and ask for a deterministic choice. See
//! `docs/superpowers/specs/2026-09-25-model-routing-design.md`.

pub mod classifier;
pub mod config;
#[cfg(feature = "discovery")]
pub mod discovery;
pub mod merge;
pub mod profile;
pub mod requirements;
pub mod select;

pub use config::{
    ConfigIssue, DefaultEndpoint, EndpointConfig, EndpointKind, RosterEntry, RoutingConfig,
};
pub use merge::{DiscoveredModel, DiscoveryOutcome, DiscoveryReport, merge};
pub use profile::{
    Availability, Capability, CapabilitySet, EndpointRef, Locality, ModelKey, ModelProfile,
    ProfileSource, Strength, Tier,
};
pub use requirements::{
    HardNeed, HardRequirements, Requirements, RequirementsBuilder, SoftPreferences, TaskKind,
    TaskOverride, TaskOverrides,
};
pub use select::{Decision, NoCandidate, Policy, ReasonPart, Unmet, select};
