//! Residency signals (feature `discovery`): which models are loaded, and
//! the host's VRAM. Cheap endpoints that products poll on a short TTL —
//! never per request. Never fails the caller: an unreachable or
//! unparseable source contributes nothing.

use serde::Deserialize;

use std::collections::HashSet;

use tokio::time::Instant;

use super::{FetchError, REQUEST_TIMEOUT, fetch_json};
use crate::config::{EndpointConfig, EndpointKind, RoutingConfig};
use crate::profile::{EndpointRef, Locality, ModelKey};
use crate::residency::{ModelResidency, ResidencySnapshot, SlotPressure, Vram};

/// A product's `aivyx-broker`. Brokers front the product's *default*
/// backend, so what the broker reports is keyed to `endpoint`. A broker
/// reporting exactly one model, loaded, also marks `endpoint` resident.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrokerSource {
    pub endpoint: EndpointRef,
    pub base_url: String,
}

/// `config`'s `[routing.endpoints]`, in name order, for [`collect`].
/// Products append their own default endpoint.
pub fn endpoints_of(config: &RoutingConfig) -> Vec<(EndpointRef, EndpointConfig)> {
    config
        .endpoints
        .iter()
        .map(|(name, endpoint)| (EndpointRef::new(name.as_str()), endpoint.clone()))
        .collect()
}

/// One residency snapshot from every source: Ollama, llama.cpp-router and
/// Lemonade endpoints (never one whose effective locality is `Cloud`),
/// then the broker (whose VRAM figure wins), then `vram_bytes`
/// as the total when nothing reported VRAM.
pub async fn collect(
    endpoints: &[(EndpointRef, EndpointConfig)],
    broker: Option<&BrokerSource>,
    vram_bytes: Option<u64>,
    client: &reqwest::Client,
) -> ResidencySnapshot {
    let mut snap = ResidencySnapshot::default();
    for (name, endpoint) in endpoints {
        // Never poll anything the crate calls cloud.
        if endpoint.effective_locality() == Locality::Cloud {
            continue;
        }
        let Some(base) = endpoint.base_url() else {
            continue;
        };
        let base = base.trim_end_matches('/');
        let models = match endpoint.kind {
            EndpointKind::Ollama => ollama(base, client).await,
            EndpointKind::LlamaRouter => llama_router(base, client).await,
            EndpointKind::Lemonade => lemonade(base, client).await,
            // Nothing to ask: single-model OpenAI-compatible servers and
            // cloud endpoints report no residency.
            EndpointKind::OpenaiCompat | EndpointKind::Anthropic | EndpointKind::Openai => {
                continue;
            }
        };
        if let Ok(models) = models {
            for (id, residency) in models {
                snap.models.insert(
                    ModelKey {
                        endpoint: name.clone(),
                        id,
                    },
                    residency,
                );
            }
        }
    }
    if let Some(broker) = broker
        && let Ok(report) = broker_report(&broker.base_url, client).await
    {
        // A single loaded model is a single-model server: whatever the
        // product calls its model, it is resident.
        if let [only] = report.models.as_slice()
            && only.loaded
        {
            snap.resident_endpoints.insert(broker.endpoint.clone());
        }
        for model in report.models {
            let residency = if model.loaded {
                ModelResidency::Loaded { vram_bytes: None }
            } else {
                ModelResidency::NotLoaded { size_bytes: None }
            };
            snap.models.insert(
                ModelKey {
                    endpoint: broker.endpoint.clone(),
                    id: model.id,
                },
                residency,
            );
        }
        snap.vram = report.vram.map(|v| Vram {
            total_bytes: v.total_bytes,
            used_bytes: v.used_bytes,
        });
        if let Some(slots) = report.slots {
            snap.slots.insert(
                broker.endpoint.clone(),
                SlotPressure {
                    busy: slots.busy,
                    total: slots.total,
                },
            );
        }
    }
    if snap.vram.is_none()
        && let Some(total_bytes) = vram_bytes
    {
        let used_bytes = snap
            .models
            .values()
            .filter_map(|m| match m {
                ModelResidency::Loaded { vram_bytes } => *vram_bytes,
                ModelResidency::NotLoaded { .. } => None,
            })
            .fold(0u64, u64::saturating_add);
        snap.vram = Some(Vram {
            total_bytes,
            used_bytes,
        });
    }
    snap
}

/// One residency request: bounded by [`REQUEST_TIMEOUT`], refusing
/// redirects and oversized bodies like discovery does.
async fn get_json<T: serde::de::DeserializeOwned>(
    url: String,
    client: &reqwest::Client,
) -> Result<T, FetchError> {
    fetch_json(client.get(url), Instant::now() + REQUEST_TIMEOUT).await
}

#[derive(Deserialize)]
struct OllamaList {
    models: Vec<OllamaModel>,
}

#[derive(Deserialize)]
struct OllamaModel {
    name: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    size_vram: Option<u64>,
}

/// `/api/ps` models are loaded (holding `size_vram`); every other
/// `/api/tags` model needs a load of `size`.
async fn ollama(
    base: &str,
    client: &reqwest::Client,
) -> Result<Vec<(String, ModelResidency)>, FetchError> {
    let running: OllamaList = get_json(format!("{base}/api/ps"), client).await?;
    let tags: OllamaList = get_json(format!("{base}/api/tags"), client).await?;
    Ok(ollama_residency(running, tags))
}

fn ollama_residency(running: OllamaList, tags: OllamaList) -> Vec<(String, ModelResidency)> {
    let mut out: Vec<(String, ModelResidency)> = running
        .models
        .into_iter()
        .map(|m| {
            let residency = ModelResidency::Loaded {
                vram_bytes: m.size_vram,
            };
            (m.name, residency)
        })
        .collect();
    let mut seen: HashSet<String> = out.iter().map(|(id, _)| id.clone()).collect();
    for model in tags.models {
        if seen.insert(model.name.clone()) {
            let residency = ModelResidency::NotLoaded {
                size_bytes: model.size,
            };
            out.push((model.name, residency));
        }
    }
    with_untagged_aliases(out)
}

/// Ollama reports a tagless model as `name:latest`, but operators often
/// write the bare `name` (it's what `ollama run` accepts), so each
/// `name:latest` also answers to `name` unless that id is already listed.
/// A loaded alias carries no `vram_bytes`: it's the same model, and its
/// VRAM is already counted once (as evictable) on the `:latest` entry.
fn with_untagged_aliases(
    mut models: Vec<(String, ModelResidency)>,
) -> Vec<(String, ModelResidency)> {
    let ids: HashSet<&str> = models.iter().map(|(id, _)| id.as_str()).collect();
    let aliases: Vec<(String, ModelResidency)> = models
        .iter()
        .filter_map(|(id, residency)| {
            let untagged = id.strip_suffix(":latest")?;
            if ids.contains(untagged) {
                return None;
            }
            let residency = match residency {
                ModelResidency::Loaded { .. } => ModelResidency::Loaded { vram_bytes: None },
                not_loaded => *not_loaded,
            };
            Some((untagged.to_string(), residency))
        })
        .collect();
    models.extend(aliases);
    models
}

#[derive(Deserialize)]
struct RouterModels {
    data: Vec<RouterModel>,
}

#[derive(Deserialize)]
struct RouterModel {
    id: String,
    #[serde(default)]
    status: Option<RouterStatus>,
}

#[derive(Deserialize)]
struct RouterStatus {
    value: String,
}

/// llama.cpp router mode `GET /models`: `loaded`/`loading` ⇒ loaded,
/// `unloaded`/`sleeping` ⇒ not loaded, no status ⇒ loaded (a
/// single-model server), anything else skipped.
async fn llama_router(
    base: &str,
    client: &reqwest::Client,
) -> Result<Vec<(String, ModelResidency)>, FetchError> {
    let models: RouterModels = get_json(format!("{base}/models"), client).await?;
    Ok(models
        .data
        .into_iter()
        .filter_map(|m| {
            let residency = match m.status.as_ref().map(|s| s.value.as_str()) {
                None | Some("loaded" | "loading") => ModelResidency::Loaded { vram_bytes: None },
                Some("unloaded" | "sleeping") => ModelResidency::NotLoaded { size_bytes: None },
                Some(_) => return None,
            };
            Some((m.id, residency))
        })
        .collect())
}

#[derive(Deserialize)]
struct LemonadeHealth {
    #[serde(default)]
    all_models_loaded: Vec<LemonadeLoaded>,
}

#[derive(Deserialize)]
struct LemonadeLoaded {
    model_name: String,
    /// Everything listed is loaded; default so a release that drops the
    /// field still parses.
    #[serde(default = "loaded_by_default")]
    loaded: bool,
}

fn loaded_by_default() -> bool {
    true
}

#[derive(Deserialize)]
struct LemonadeModels {
    data: Vec<LemonadeModel>,
}

#[derive(Deserialize)]
struct LemonadeModel {
    id: String,
    #[serde(default)]
    downloaded: bool,
    /// GB; absent means the size is unknown, not zero.
    #[serde(default)]
    size: Option<f64>,
}

/// Lemonade holds one LLM at a time. `/v1/health` says which downloaded
/// model (if any) is loaded; `/v1/models` supplies the size of every other
/// downloaded model, needing a load. A `/v1/models` failure still leaves
/// the loaded model's entry, from health alone.
async fn lemonade(
    base: &str,
    client: &reqwest::Client,
) -> Result<Vec<(String, ModelResidency)>, FetchError> {
    let health: LemonadeHealth = get_json(super::lemonade::api_url(base, "health"), client).await?;
    let mut out: Vec<(String, ModelResidency)> = health
        .all_models_loaded
        .into_iter()
        .filter(|m| m.loaded)
        .map(|m| (m.model_name, ModelResidency::Loaded { vram_bytes: None }))
        .collect();
    if let Ok(models) =
        get_json::<LemonadeModels>(super::lemonade::api_url(base, "models"), client).await
    {
        let mut seen: HashSet<String> = out.iter().map(|(id, _)| id.clone()).collect();
        for model in models.data {
            if model.downloaded && seen.insert(model.id.clone()) {
                let size_bytes = model.size.map(|gb| (gb * 1e9).round() as u64);
                out.push((model.id, ModelResidency::NotLoaded { size_bytes }));
            }
        }
    }
    Ok(out)
}

#[derive(Deserialize)]
struct BrokerReport {
    models: Vec<BrokerModel>,
    #[serde(default)]
    vram: Option<BrokerVram>,
    #[serde(default)]
    slots: Option<BrokerSlots>,
}

#[derive(Deserialize)]
struct BrokerModel {
    id: String,
    loaded: bool,
}

#[derive(Deserialize)]
struct BrokerVram {
    total_bytes: u64,
    used_bytes: u64,
}

#[derive(Deserialize)]
struct BrokerSlots {
    busy: u32,
    total: u32,
}

async fn broker_report(base: &str, client: &reqwest::Client) -> Result<BrokerReport, FetchError> {
    get_json(
        format!("{}/v1/aivyx/residency", base.trim_end_matches('/')),
        client,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EndpointKind;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const TAGS: &str = include_str!("../../tests/fixtures/ollama_tags.json");
    const PS: &str = include_str!("../../tests/fixtures/ollama_ps.json");
    const ROUTER: &str = include_str!("../../tests/fixtures/llama_router_models_status.json");
    const BROKER: &str = include_str!("../../tests/fixtures/broker_residency.json");
    const LEMONADE_MODELS: &str = include_str!("../../tests/fixtures/lemonade_models.json");
    const LEMONADE_HEALTH: &str = include_str!("../../tests/fixtures/lemonade_health.json");

    fn json_response(body: &str) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_raw(body.to_owned(), "application/json")
    }

    async fn serve(route: &str, body: &str) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(route))
            .respond_with(json_response(body))
            .mount(&server)
            .await;
        server
    }

    fn endpoint(kind: EndpointKind, base: &str) -> EndpointConfig {
        EndpointConfig {
            kind,
            base_url: Some(base.to_string()),
            locality: None,
        }
    }

    fn key(endpoint: &str, id: &str) -> ModelKey {
        ModelKey {
            endpoint: EndpointRef::new(endpoint),
            id: id.into(),
        }
    }

    #[tokio::test]
    async fn ollama_loaded_models_hold_vram_and_the_rest_need_loading() {
        let server = serve("/api/ps", PS).await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(json_response(TAGS))
            .mount(&server)
            .await;
        let snap = collect(
            &[(
                EndpointRef::new("ollama"),
                endpoint(EndpointKind::Ollama, &server.uri()),
            )],
            None,
            None,
            &reqwest::Client::new(),
        )
        .await;
        assert_eq!(
            snap.models[&key("ollama", "qwen3-coder:30b")],
            ModelResidency::Loaded {
                vram_bytes: Some(19_971_883_008)
            }
        );
        assert_eq!(
            snap.models[&key("ollama", "llava:13b")],
            ModelResidency::NotLoaded {
                size_bytes: Some(8_011_256_494)
            }
        );
        assert_eq!(snap.vram, None);
    }

    #[test]
    fn ollama_dedup_scales_to_a_huge_tag_list() {
        let list = |n: usize| OllamaList {
            models: (0..n)
                .map(|i| OllamaModel {
                    name: format!("model-{i}:latest"),
                    size: Some(1),
                    size_vram: Some(1),
                })
                .collect(),
        };
        let started = std::time::Instant::now();
        let out = ollama_residency(list(20_000), list(40_000));
        // 40k ids plus 40k untagged aliases.
        assert_eq!(out.len(), 80_000);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "took {:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn endpoints_whose_address_is_cloud_are_never_polled() {
        let server = serve("/api/ps", PS).await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(json_response(TAGS))
            .mount(&server)
            .await;
        let mut forced = endpoint(EndpointKind::Ollama, &server.uri());
        forced.locality = Some(crate::profile::Locality::Cloud);
        let snap = collect(
            &[
                (EndpointRef::new("forced"), forced),
                (
                    EndpointRef::new("hosted"),
                    endpoint(EndpointKind::Lemonade, "http://203.0.113.7:9/api"),
                ),
            ],
            None,
            None,
            &reqwest::Client::new(),
        )
        .await;
        assert!(snap.models.is_empty(), "{:?}", snap.models);
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn huge_vram_figures_saturate_instead_of_overflowing() {
        let server = serve(
            "/api/ps",
            r#"{"models":[{"name":"a:1b","size_vram":18446744073709551615},{"name":"b:1b","size_vram":18446744073709551615}]}"#,
        )
        .await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(json_response(r#"{"models":[]}"#))
            .mount(&server)
            .await;
        let snap = collect(
            &[(
                EndpointRef::new("ollama"),
                endpoint(EndpointKind::Ollama, &server.uri()),
            )],
            None,
            Some(24 << 30),
            &reqwest::Client::new(),
        )
        .await;
        assert_eq!(snap.vram.map(|v| v.used_bytes), Some(u64::MAX));
        assert_eq!(snap.available_vram(), Some(24 << 30));
    }

    #[tokio::test]
    async fn oversized_and_redirected_responses_contribute_nothing() {
        let big = format!(
            r#"{{"models":[{{"name":"pad","size":1,"digest":"{}"}}]}}"#,
            "x".repeat(9 << 20)
        );
        let oversized = serve("/api/ps", r#"{"models":[]}"#).await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(json_response(&big))
            .mount(&oversized)
            .await;
        let redirected = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(302).insert_header("location", "/elsewhere"))
            .mount(&redirected)
            .await;
        Mock::given(method("GET"))
            .and(path("/elsewhere"))
            .respond_with(json_response(ROUTER))
            .mount(&redirected)
            .await;
        let snap = collect(
            &[
                (
                    EndpointRef::new("ollama"),
                    endpoint(EndpointKind::Ollama, &oversized.uri()),
                ),
                (
                    EndpointRef::new("router"),
                    endpoint(EndpointKind::LlamaRouter, &redirected.uri()),
                ),
            ],
            None,
            None,
            &reqwest::Client::new(),
        )
        .await;
        assert!(snap.models.is_empty(), "{:?}", snap.models);
    }

    /// Ollama reports a tagless model as `name:latest`, while operators
    /// often write `name` (in `[backend] model`, `[agent] model` or a
    /// roster entry). Both keys must resolve, and the VRAM counts once.
    #[tokio::test]
    async fn ollama_latest_models_also_answer_to_their_untagged_name() {
        let server = serve(
            "/api/ps",
            r#"{"models":[{"name":"llama3.2:latest","size":2000000000,"size_vram":2000000000}]}"#,
        )
        .await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(json_response(
                r#"{"models":[{"name":"llama3.2:latest","size":2000000000},{"name":"phi4:latest","size":9000000000},{"name":"qwen3:8b","size":5000000000}]}"#,
            ))
            .mount(&server)
            .await;
        let snap = collect(
            &[(
                EndpointRef::new("ollama"),
                endpoint(EndpointKind::Ollama, &server.uri()),
            )],
            None,
            Some(24 << 30),
            &reqwest::Client::new(),
        )
        .await;
        assert_eq!(
            snap.models[&key("ollama", "llama3.2")],
            ModelResidency::Loaded { vram_bytes: None },
            "the alias carries no VRAM of its own"
        );
        assert_eq!(
            snap.models[&key("ollama", "phi4")],
            ModelResidency::NotLoaded {
                size_bytes: Some(9_000_000_000)
            }
        );
        assert!(
            !snap.models.contains_key(&key("ollama", "qwen3")),
            "only :latest gets an alias"
        );
        // used = what loaded models hold, counted once.
        assert_eq!(snap.vram.map(|v| v.used_bytes), Some(2_000_000_000));
    }

    #[test]
    fn an_already_listed_bare_name_is_left_alone() {
        let own = ModelResidency::NotLoaded {
            size_bytes: Some(1),
        };
        let latest = ModelResidency::Loaded {
            vram_bytes: Some(7),
        };
        let got = with_untagged_aliases(vec![
            ("m".to_string(), own),
            ("m:latest".to_string(), latest),
        ]);
        assert_eq!(
            got,
            vec![("m".to_string(), own), ("m:latest".to_string(), latest)]
        );
    }

    #[tokio::test]
    async fn llama_router_statuses_map_to_residency() {
        let server = serve("/models", ROUTER).await;
        let snap = collect(
            &[(
                EndpointRef::new("gpu"),
                endpoint(EndpointKind::LlamaRouter, &server.uri()),
            )],
            None,
            None,
            &reqwest::Client::new(),
        )
        .await;
        let loaded = ModelResidency::Loaded { vram_bytes: None };
        let cold = ModelResidency::NotLoaded { size_bytes: None };
        assert_eq!(snap.models[&key("gpu", "gemma-3-4b")], loaded);
        assert_eq!(snap.models[&key("gpu", "qwen3-8b")], loaded);
        assert_eq!(snap.models[&key("gpu", "phi-4")], cold);
        assert_eq!(snap.models[&key("gpu", "mistral-7b")], cold);
        assert_eq!(snap.models[&key("gpu", "broken")], cold);
        assert!(
            !snap.models.contains_key(&key("gpu", "odd")),
            "unknown status is skipped"
        );
    }

    #[tokio::test]
    async fn a_model_without_status_is_a_single_model_server_and_loaded() {
        let server = serve("/models", r#"{"data":[{"id":"only"}]}"#).await;
        let snap = collect(
            &[(
                EndpointRef::new("gpu"),
                endpoint(EndpointKind::LlamaRouter, &server.uri()),
            )],
            None,
            None,
            &reqwest::Client::new(),
        )
        .await;
        assert_eq!(
            snap.models[&key("gpu", "only")],
            ModelResidency::Loaded { vram_bytes: None }
        );
    }

    async fn serve_lemonade(health: &str, models: Option<&str>) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/health"))
            .respond_with(json_response(health))
            .mount(&server)
            .await;
        if let Some(models) = models {
            Mock::given(method("GET"))
                .and(path("/v1/models"))
                .respond_with(json_response(models))
                .mount(&server)
                .await;
        }
        server
    }

    #[tokio::test]
    async fn lemonade_loaded_model_is_loaded_and_the_rest_need_loading() {
        let server = serve_lemonade(LEMONADE_HEALTH, Some(LEMONADE_MODELS)).await;
        let snap = collect(
            &[(
                EndpointRef::new("lemonade"),
                endpoint(EndpointKind::Lemonade, &server.uri()),
            )],
            None,
            None,
            &reqwest::Client::new(),
        )
        .await;
        assert_eq!(
            snap.models[&key("lemonade", "Qwen3-4B-Instruct-2507-GGUF")],
            ModelResidency::Loaded { vram_bytes: None }
        );
        assert_eq!(
            snap.models[&key("lemonade", "Qwen3.5-9B-GGUF")],
            ModelResidency::NotLoaded {
                size_bytes: Some(6_410_000_000)
            }
        );
        assert!(
            !snap
                .models
                .contains_key(&key("lemonade", "Gemma-3-1B-GGUF")),
            "an undownloaded model must never appear"
        );
        assert!(
            snap.resident_endpoints.is_empty(),
            "a Lemonade endpoint is never marked resident"
        );
    }

    #[tokio::test]
    async fn lemonade_with_nothing_loaded_makes_every_model_notloaded() {
        let health =
            r#"{"model_loaded":null,"all_models_loaded":[],"max_models":{"llm":1},"status":"ok"}"#;
        let server = serve_lemonade(health, Some(LEMONADE_MODELS)).await;
        let snap = collect(
            &[(
                EndpointRef::new("lemonade"),
                endpoint(EndpointKind::Lemonade, &server.uri()),
            )],
            None,
            None,
            &reqwest::Client::new(),
        )
        .await;
        assert_eq!(
            snap.models[&key("lemonade", "Qwen3-4B-Instruct-2507-GGUF")],
            ModelResidency::NotLoaded {
                size_bytes: Some(2_330_000_000)
            }
        );
        assert_eq!(
            snap.models[&key("lemonade", "Qwen3.5-9B-GGUF")],
            ModelResidency::NotLoaded {
                size_bytes: Some(6_410_000_000)
            }
        );
    }

    #[tokio::test]
    async fn lemonade_residency_works_on_a_base_ending_in_v1_and_without_a_loaded_field() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/health"))
            .respond_with(json_response(
                r#"{"all_models_loaded":[{"model_name":"Qwen3-4B-Instruct-2507-GGUF"}]}"#,
            ))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(json_response(LEMONADE_MODELS))
            .mount(&server)
            .await;
        let snap = collect(
            &[(
                EndpointRef::new("lemonade"),
                endpoint(EndpointKind::Lemonade, &format!("{}/api/v1", server.uri())),
            )],
            None,
            None,
            &reqwest::Client::new(),
        )
        .await;
        assert_eq!(
            snap.models[&key("lemonade", "Qwen3-4B-Instruct-2507-GGUF")],
            ModelResidency::Loaded { vram_bytes: None }
        );
        assert_eq!(
            snap.models[&key("lemonade", "Qwen3.5-9B-GGUF")],
            ModelResidency::NotLoaded {
                size_bytes: Some(6_410_000_000)
            }
        );
    }

    #[tokio::test]
    async fn a_lemonade_model_without_a_size_has_an_unknown_size() {
        let health = r#"{"status":"ok"}"#;
        let models = r#"{"data":[{"id":"NoSize-GGUF","downloaded":true}]}"#;
        let server = serve_lemonade(health, Some(models)).await;
        let snap = collect(
            &[(
                EndpointRef::new("lemonade"),
                endpoint(EndpointKind::Lemonade, &server.uri()),
            )],
            None,
            None,
            &reqwest::Client::new(),
        )
        .await;
        assert_eq!(
            snap.models[&key("lemonade", "NoSize-GGUF")],
            ModelResidency::NotLoaded { size_bytes: None }
        );
    }

    #[tokio::test]
    async fn a_lemonade_models_failure_with_good_health_still_marks_the_loaded_one_loaded() {
        // No /v1/models mock: wiremock answers 404.
        let server = serve_lemonade(LEMONADE_HEALTH, None).await;
        let snap = collect(
            &[(
                EndpointRef::new("lemonade"),
                endpoint(EndpointKind::Lemonade, &server.uri()),
            )],
            None,
            None,
            &reqwest::Client::new(),
        )
        .await;
        assert_eq!(
            snap.models[&key("lemonade", "Qwen3-4B-Instruct-2507-GGUF")],
            ModelResidency::Loaded { vram_bytes: None }
        );
        assert!(
            !snap
                .models
                .contains_key(&key("lemonade", "Qwen3.5-9B-GGUF"))
        );
    }

    #[tokio::test]
    async fn the_broker_reports_the_default_endpoint_vram_and_slots() {
        let server = serve("/v1/aivyx/residency", BROKER).await;
        let broker = BrokerSource {
            endpoint: EndpointRef::new("backend"),
            base_url: server.uri(),
        };
        // The broker's VRAM wins over the operator figure.
        let snap = collect(&[], Some(&broker), Some(1), &reqwest::Client::new()).await;
        assert_eq!(
            snap.models[&key("backend", "qwen3-8b")],
            ModelResidency::Loaded { vram_bytes: None }
        );
        assert_eq!(
            snap.models[&key("backend", "gemma-3-4b")],
            ModelResidency::NotLoaded { size_bytes: None }
        );
        assert_eq!(
            snap.vram,
            Some(Vram {
                total_bytes: 25_769_803_776,
                used_bytes: 9_663_676_416
            })
        );
        assert_eq!(
            snap.slots[&EndpointRef::new("backend")],
            SlotPressure { busy: 1, total: 2 }
        );
    }

    fn broker_at(server: &MockServer) -> BrokerSource {
        BrokerSource {
            endpoint: EndpointRef::new("backend"),
            base_url: server.uri(),
        }
    }

    #[tokio::test]
    async fn a_broker_with_one_loaded_model_marks_its_endpoint_resident() {
        let body = r#"{"models":[{"id":"/models/qwen3-8b-Q4_K_M.gguf","loaded":true}],"vram":null,"slots":{"busy":0,"total":1}}"#;
        let server = serve("/v1/aivyx/residency", body).await;
        let snap = collect(
            &[],
            Some(&broker_at(&server)),
            None,
            &reqwest::Client::new(),
        )
        .await;
        assert!(
            snap.resident_endpoints
                .contains(&EndpointRef::new("backend"))
        );
        // The per-id entry is still recorded.
        assert_eq!(
            snap.models[&key("backend", "/models/qwen3-8b-Q4_K_M.gguf")],
            ModelResidency::Loaded { vram_bytes: None }
        );
    }

    #[tokio::test]
    async fn a_multi_model_broker_does_not_mark_its_endpoint_resident() {
        let server = serve("/v1/aivyx/residency", BROKER).await;
        let snap = collect(
            &[],
            Some(&broker_at(&server)),
            None,
            &reqwest::Client::new(),
        )
        .await;
        assert!(snap.resident_endpoints.is_empty());
    }

    #[tokio::test]
    async fn a_single_unloaded_broker_model_does_not_mark_its_endpoint_resident() {
        let body =
            r#"{"models":[{"id":"only","loaded":false}],"vram":null,"slots":{"busy":0,"total":1}}"#;
        let server = serve("/v1/aivyx/residency", body).await;
        let snap = collect(
            &[],
            Some(&broker_at(&server)),
            None,
            &reqwest::Client::new(),
        )
        .await;
        assert!(snap.resident_endpoints.is_empty());
    }

    #[tokio::test]
    async fn the_broker_body_without_upstream_or_gpu_parses() {
        let body = r#"{"models":[],"vram":null,"slots":{"busy":0,"total":2}}"#;
        let server = serve("/v1/aivyx/residency", body).await;
        let snap = collect(
            &[],
            Some(&broker_at(&server)),
            None,
            &reqwest::Client::new(),
        )
        .await;
        assert_eq!(snap.vram, None);
        assert!(snap.models.is_empty());
        assert!(snap.resident_endpoints.is_empty());
        assert_eq!(
            snap.slots[&EndpointRef::new("backend")],
            SlotPressure { busy: 0, total: 2 }
        );
    }

    #[tokio::test]
    async fn a_broker_body_without_slots_parses() {
        let body = r#"{"models":[{"id":"a","loaded":false}],"vram":null}"#;
        let server = serve("/v1/aivyx/residency", body).await;
        let snap = collect(
            &[],
            Some(&broker_at(&server)),
            None,
            &reqwest::Client::new(),
        )
        .await;
        assert_eq!(
            snap.models[&key("backend", "a")],
            ModelResidency::NotLoaded { size_bytes: None }
        );
        assert!(snap.slots.is_empty());
    }

    #[tokio::test]
    async fn without_a_broker_figure_vram_bytes_supplies_the_total() {
        let server = serve("/api/ps", PS).await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(json_response(TAGS))
            .mount(&server)
            .await;
        let snap = collect(
            &[(
                EndpointRef::new("ollama"),
                endpoint(EndpointKind::Ollama, &server.uri()),
            )],
            None,
            Some(24 << 30),
            &reqwest::Client::new(),
        )
        .await;
        // used = what the loaded models hold.
        assert_eq!(
            snap.vram,
            Some(Vram {
                total_bytes: 24 << 30,
                used_bytes: 19_971_883_008
            })
        );
    }

    #[tokio::test]
    async fn unreachable_sources_contribute_nothing() {
        let broker = BrokerSource {
            endpoint: EndpointRef::new("backend"),
            base_url: "http://127.0.0.1:1".to_string(),
        };
        let snap = collect(
            &[
                (
                    EndpointRef::new("ollama"),
                    endpoint(EndpointKind::Ollama, "http://127.0.0.1:1"),
                ),
                (
                    EndpointRef::new("compat"),
                    endpoint(EndpointKind::OpenaiCompat, "http://127.0.0.1:1"),
                ),
                (
                    EndpointRef::new("lemonade"),
                    endpoint(EndpointKind::Lemonade, "http://127.0.0.1:1"),
                ),
                (
                    EndpointRef::new("cloud"),
                    EndpointConfig {
                        kind: EndpointKind::Anthropic,
                        base_url: None,
                        locality: None,
                    },
                ),
            ],
            Some(&broker),
            None,
            &reqwest::Client::new(),
        )
        .await;
        assert_eq!(snap, ResidencySnapshot::default());
    }

    #[test]
    fn endpoints_of_lists_the_configured_endpoints_in_name_order() {
        let mut config = RoutingConfig::default();
        config
            .endpoints
            .insert("b".into(), endpoint(EndpointKind::Ollama, "http://b"));
        config
            .endpoints
            .insert("a".into(), endpoint(EndpointKind::LlamaRouter, "http://a"));
        let names: Vec<String> = endpoints_of(&config)
            .into_iter()
            .map(|(e, _)| e.to_string())
            .collect();
        assert_eq!(names, ["a", "b"]);
    }
}
