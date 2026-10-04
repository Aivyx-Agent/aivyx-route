//! Generic OpenAI-compatible servers: `GET /v1/models` yields ids only, so
//! every capability is unknown.

use serde::Deserialize;

use tokio::time::Instant;

use super::{FetchError, fetch_json};
use crate::merge::DiscoveredModel;
use crate::profile::{Capability, CapabilitySet};

#[derive(Deserialize)]
struct Models {
    data: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    id: String,
}

/// Bases conventionally either include `/v1` (Jan: `http://localhost:1337/v1`)
/// or not (`llama-server`: `http://localhost:8080`).
fn models_url(base: &str) -> String {
    if base.ends_with("/v1") {
        format!("{base}/models")
    } else {
        format!("{base}/v1/models")
    }
}

pub(super) async fn discover(
    base: &str,
    client: &reqwest::Client,
    deadline: Instant,
) -> Result<Vec<DiscoveredModel>, FetchError> {
    let models: Models = fetch_json(client.get(models_url(base)), deadline).await?;
    Ok(models
        .data
        .into_iter()
        .map(|entry| DiscoveredModel {
            id: entry.id,
            capabilities: CapabilitySet::new(),
            unknown_capabilities: Capability::ALL.into_iter().collect(),
            context_window: None,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn models_url_handles_bases_with_and_without_v1() {
        assert_eq!(models_url("http://h:1337/v1"), "http://h:1337/v1/models");
        assert_eq!(models_url("http://h:8080"), "http://h:8080/v1/models");
    }

    #[tokio::test]
    async fn lists_ids_with_no_capabilities() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                include_str!("../../tests/fixtures/openai_models.json"),
                "application/json",
            ))
            .mount(&server)
            .await;
        for base in [server.uri(), format!("{}/v1", server.uri())] {
            let models = discover(
                &base,
                &reqwest::Client::new(),
                crate::discovery::far_deadline(),
            )
            .await
            .unwrap();
            assert_eq!(models.len(), 1, "{base}");
            assert_eq!(models[0].id, "qwen3-8b-q4_k_m.gguf");
            assert!(models[0].capabilities.is_empty());
            assert_eq!(
                models[0].unknown_capabilities,
                Capability::ALL.into_iter().collect::<CapabilitySet>()
            );
        }
    }
}
