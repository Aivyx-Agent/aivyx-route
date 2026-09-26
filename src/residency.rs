//! Model residency: which models are loaded, and how much GPU memory a new
//! load could use. Data only — `discovery::residency` fills it, `select`
//! scores it. Soft: it never affects hard filtering, stickiness or pins.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::profile::{EndpointRef, ModelKey, ModelProfile};

/// Residency costs are in quarter tier-steps: the largest equals one.
pub const TIER_STEP: u16 = 4;

/// One model's load state, as its backend reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelResidency {
    /// In memory (or loading). `vram_bytes`: what it holds, if reported.
    Loaded { vram_bytes: Option<u64> },
    /// Would need a load first. `size_bytes`: its size, if reported.
    NotLoaded { size_bytes: Option<u64> },
}

/// The host's GPU memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vram {
    pub total_bytes: u64,
    pub used_bytes: u64,
}

/// Busy and total inference slots on one endpoint. Reported, not scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotPressure {
    pub busy: u32,
    pub total: u32,
}

/// Everything known about residency at one moment. Empty (the default)
/// leaves every decision exactly as without residency.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidencySnapshot {
    pub models: BTreeMap<ModelKey, ModelResidency>,
    /// Endpoints whose every model counts as loaded when it has no entry
    /// of its own (a single-model server: whatever the product calls its
    /// model, it is resident).
    pub resident_endpoints: BTreeSet<EndpointRef>,
    pub vram: Option<Vram>,
    pub slots: BTreeMap<EndpointRef, SlotPressure>,
}

/// What residency said about a chosen model, for its reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidencyNote {
    Loaded,
    NeedsLoad,
    WontFit,
}

impl ResidencySnapshot {
    /// VRAM a new load could use after evicting every loaded model this
    /// snapshot knows about; `None` without a VRAM figure.
    pub fn available_vram(&self) -> Option<u64> {
        let vram = self.vram?;
        let evictable: u64 = self
            .models
            .values()
            .filter_map(|m| match m {
                ModelResidency::Loaded { vram_bytes } => *vram_bytes,
                ModelResidency::NotLoaded { .. } => None,
            })
            .sum();
        let held_elsewhere = vram.used_bytes.saturating_sub(evictable);
        Some(vram.total_bytes.saturating_sub(held_elsewhere))
    }

    /// `profile`'s residency cost (0..=[`TIER_STEP`]) and note. No entry
    /// costs 2, the same as a load of unknown size, so an empty snapshot
    /// ranks every model equally — unless its endpoint is in
    /// [`Self::resident_endpoints`], which makes it loaded. A per-model
    /// entry always wins over the endpoint signal.
    pub fn cost(&self, profile: &ModelProfile) -> (u16, Option<ResidencyNote>) {
        match self.models.get(&profile.key()) {
            None if self.resident_endpoints.contains(&profile.endpoint) => {
                (0, Some(ResidencyNote::Loaded))
            }
            None => (2, None),
            Some(ModelResidency::Loaded { .. }) => (0, Some(ResidencyNote::Loaded)),
            Some(ModelResidency::NotLoaded { size_bytes }) => {
                match (*size_bytes, self.available_vram()) {
                    (Some(size), Some(available)) if size > available => {
                        (TIER_STEP, Some(ResidencyNote::WontFit))
                    }
                    (Some(size), Some(available)) if size <= available / 4 => {
                        (1, Some(ResidencyNote::NeedsLoad))
                    }
                    (Some(size), Some(available)) if size <= available / 2 => {
                        (2, Some(ResidencyNote::NeedsLoad))
                    }
                    (Some(_), Some(_)) => (3, Some(ResidencyNote::NeedsLoad)),
                    _ => (2, Some(ResidencyNote::NeedsLoad)),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: u64 = 1 << 30;

    fn key(id: &str) -> ModelKey {
        ModelKey {
            endpoint: EndpointRef::new("local"),
            id: id.into(),
        }
    }

    fn profile(id: &str) -> ModelProfile {
        ModelProfile::new(id, EndpointRef::new("local"))
    }

    fn snapshot(models: &[(&str, ModelResidency)], vram: Option<(u64, u64)>) -> ResidencySnapshot {
        ResidencySnapshot {
            models: models.iter().map(|(id, r)| (key(id), *r)).collect(),
            vram: vram.map(|(total_bytes, used_bytes)| Vram {
                total_bytes,
                used_bytes,
            }),
            slots: BTreeMap::new(),
            resident_endpoints: BTreeSet::new(),
        }
    }

    fn with_resident_local(mut s: ResidencySnapshot) -> ResidencySnapshot {
        s.resident_endpoints.insert(EndpointRef::new("local"));
        s
    }

    #[test]
    fn available_vram_counts_loaded_models_as_evictable() {
        // 24G card, 20G used, 12G of it by a loaded model: 8G is held by
        // something else, so a load could use 16G.
        let s = snapshot(
            &[(
                "x",
                ModelResidency::Loaded {
                    vram_bytes: Some(12 * G),
                },
            )],
            Some((24 * G, 20 * G)),
        );
        assert_eq!(s.available_vram(), Some(16 * G));
        assert_eq!(snapshot(&[], None).available_vram(), None);
        // Loaded models reporting more than `used` never underflow.
        let s = snapshot(
            &[(
                "x",
                ModelResidency::Loaded {
                    vram_bytes: Some(30 * G),
                },
            )],
            Some((24 * G, 20 * G)),
        );
        assert_eq!(s.available_vram(), Some(24 * G));
    }

    #[test]
    fn cost_follows_the_quarter_step_table() {
        let not_loaded = |size| ModelResidency::NotLoaded { size_bytes: size };
        let s = snapshot(
            &[
                ("warm", ModelResidency::Loaded { vram_bytes: None }),
                ("quarter", not_loaded(Some(6 * G))),
                ("half", not_loaded(Some(12 * G))),
                ("most", not_loaded(Some(20 * G))),
                ("huge", not_loaded(Some(30 * G))),
                ("unsized", not_loaded(None)),
            ],
            Some((24 * G, 0)),
        );
        let cost = |id| s.cost(&profile(id));
        assert_eq!(cost("warm"), (0, Some(ResidencyNote::Loaded)));
        assert_eq!(cost("quarter"), (1, Some(ResidencyNote::NeedsLoad)));
        assert_eq!(cost("half"), (2, Some(ResidencyNote::NeedsLoad)));
        assert_eq!(cost("most"), (3, Some(ResidencyNote::NeedsLoad)));
        assert_eq!(cost("huge"), (TIER_STEP, Some(ResidencyNote::WontFit)));
        assert_eq!(cost("unsized"), (2, Some(ResidencyNote::NeedsLoad)));
        assert_eq!(cost("absent"), (2, None));
    }

    #[test]
    fn without_vram_a_sized_load_costs_the_middle() {
        let s = snapshot(
            &[(
                "m",
                ModelResidency::NotLoaded {
                    size_bytes: Some(30 * G),
                },
            )],
            None,
        );
        assert_eq!(s.cost(&profile("m")), (2, Some(ResidencyNote::NeedsLoad)));
    }

    #[test]
    fn a_resident_endpoint_counts_its_unlisted_models_as_loaded() {
        let s = with_resident_local(snapshot(&[], None));
        assert_eq!(
            s.cost(&profile("anything")),
            (0, Some(ResidencyNote::Loaded))
        );
        // Another endpoint is unaffected.
        let elsewhere = ModelProfile::new("anything", EndpointRef::new("other"));
        assert_eq!(s.cost(&elsewhere), (2, None));
    }

    #[test]
    fn a_per_model_entry_wins_over_a_resident_endpoint() {
        let s = with_resident_local(snapshot(
            &[(
                "big",
                ModelResidency::NotLoaded {
                    size_bytes: Some(30 * G),
                },
            )],
            Some((24 * G, 0)),
        ));
        assert_eq!(
            s.cost(&profile("big")),
            (TIER_STEP, Some(ResidencyNote::WontFit))
        );
    }

    #[test]
    fn a_resident_endpoint_adds_no_evictable_vram() {
        let plain = snapshot(&[], Some((24 * G, 20 * G)));
        let resident = with_resident_local(plain.clone());
        assert_eq!(resident.available_vram(), Some(4 * G));
        assert_eq!(resident.available_vram(), plain.available_vram());
    }
}
