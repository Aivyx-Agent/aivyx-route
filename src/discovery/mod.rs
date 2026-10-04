//! Backend discovery (feature `discovery`): ask each configured endpoint
//! which models it serves and what they can do. Runs at startup and on
//! explicit refresh only — never per request. Never fails the caller:
//! problems become `DiscoveryOutcome::Unreachable`.

mod lemonade;
mod llama_router;
mod ollama;
mod openai_compat;
pub mod residency;

use std::time::Duration;

/// The `reqwest` this crate is built against. Consumers on a different
/// reqwest major build the client as
/// `aivyx_route::discovery::reqwest::Client::new()`.
pub use reqwest;

use crate::config::{EndpointConfig, EndpointKind, RoutingConfig};
use crate::merge::{DiscoveryOutcome, DiscoveryReport};
use crate::profile::EndpointRef;

/// Timeout for each individual HTTP request made during discovery.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Probe one endpoint.
pub async fn discover(
    endpoint: &EndpointRef,
    config: &EndpointConfig,
    client: &reqwest::Client,
) -> DiscoveryReport {
    let outcome = match config.kind {
        EndpointKind::Anthropic | EndpointKind::Openai => DiscoveryOutcome::NotProbed,
        kind => match config.base_url() {
            None => DiscoveryOutcome::Unreachable(format!(
                "no base_url configured for endpoint `{endpoint}`"
            )),
            Some(base) => {
                let base = base.trim_end_matches('/');
                let result = match kind {
                    EndpointKind::Ollama => ollama::discover(base, client).await,
                    EndpointKind::LlamaRouter => llama_router::discover(base, client).await,
                    EndpointKind::Lemonade => lemonade::discover(base, client).await,
                    _ => openai_compat::discover(base, client).await,
                };
                match result {
                    Ok(models) => DiscoveryOutcome::Reached(models),
                    Err(e) => DiscoveryOutcome::Unreachable(e.to_string()),
                }
            }
        },
    };
    DiscoveryReport {
        endpoint: endpoint.clone(),
        outcome,
    }
}

/// Probe every `[routing.endpoints]` entry, in name order.
pub async fn discover_all(
    config: &RoutingConfig,
    client: &reqwest::Client,
) -> Vec<DiscoveryReport> {
    let mut reports = Vec::with_capacity(config.endpoints.len());
    for (name, endpoint) in &config.endpoints {
        reports.push(discover(&EndpointRef::new(name.clone()), endpoint, client).await);
    }
    reports
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EndpointKind;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg(kind: EndpointKind, base_url: Option<&str>) -> EndpointConfig {
        EndpointConfig {
            kind,
            base_url: base_url.map(str::to_string),
            locality: None,
        }
    }

    #[tokio::test]
    async fn cloud_endpoints_are_never_probed() {
        let client = reqwest::Client::new();
        for kind in [EndpointKind::Anthropic, EndpointKind::Openai] {
            let r = discover(
                &EndpointRef::new("cloud"),
                &cfg(kind, Some("http://127.0.0.1:9")),
                &client,
            )
            .await;
            assert_eq!(r.outcome, DiscoveryOutcome::NotProbed);
        }
    }

    #[tokio::test]
    async fn missing_base_url_is_unreachable_with_a_reason() {
        let client = reqwest::Client::new();
        let r = discover(
            &EndpointRef::new("jan"),
            &cfg(EndpointKind::OpenaiCompat, None),
            &client,
        )
        .await;
        match r.outcome {
            DiscoveryOutcome::Unreachable(why) => assert!(why.contains("no base_url"), "{why}"),
            other => panic!("expected Unreachable, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn connection_refused_is_unreachable() {
        let client = reqwest::Client::new();
        let r = discover(
            &EndpointRef::new("down"),
            &cfg(EndpointKind::Ollama, Some("http://127.0.0.1:9")),
            &client,
        )
        .await;
        assert!(
            matches!(r.outcome, DiscoveryOutcome::Unreachable(_)),
            "{r:?}"
        );
        assert_eq!(r.endpoint, EndpointRef::new("down"));
    }

    #[tokio::test]
    async fn trailing_slash_on_base_url_is_tolerated() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                include_str!("../../tests/fixtures/llama_router_models.json"),
                "application/json",
            ))
            .mount(&server)
            .await;
        let base = format!("{}/", server.uri());
        let r = discover(
            &EndpointRef::new("router"),
            &cfg(EndpointKind::LlamaRouter, Some(&base)),
            &reqwest::Client::new(),
        )
        .await;
        assert!(
            matches!(r.outcome, DiscoveryOutcome::Reached(ref m) if m.len() == 2),
            "{r:?}"
        );
    }

    #[tokio::test]
    async fn discover_all_reports_every_endpoint_in_name_order() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                include_str!("../../tests/fixtures/openai_models.json"),
                "application/json",
            ))
            .mount(&server)
            .await;
        let mut config = RoutingConfig::default();
        config
            .endpoints
            .insert("z-cloud".into(), cfg(EndpointKind::Anthropic, None));
        config.endpoints.insert(
            "a-local".into(),
            cfg(EndpointKind::OpenaiCompat, Some(&server.uri())),
        );
        let reports = discover_all(&config, &reqwest::Client::new()).await;
        let names: Vec<&str> = reports.iter().map(|r| r.endpoint.as_str()).collect();
        assert_eq!(names, vec!["a-local", "z-cloud"]);
        assert!(
            matches!(reports[0].outcome, DiscoveryOutcome::Reached(ref m) if m[0].id == "qwen3-8b-q4_k_m.gguf")
        );
        assert_eq!(reports[1].outcome, DiscoveryOutcome::NotProbed);
    }

    #[tokio::test]
    async fn reqwest_is_re_exported_for_consumers() {
        let client = crate::discovery::reqwest::Client::new();
        assert!(
            discover_all(&RoutingConfig::default(), &client)
                .await
                .is_empty()
        );
    }
}
