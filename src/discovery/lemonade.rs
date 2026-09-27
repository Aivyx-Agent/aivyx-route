//! Lemonade Server: `GET /v1/models` lists the catalog (downloaded models
//! only); its `labels` map onto capabilities. Lemonade never reports
//! thinking or audio support, so those stay unknown regardless of labels.

use serde::Deserialize;

use super::REQUEST_TIMEOUT;
use crate::merge::DiscoveredModel;
use crate::profile::{Capability, CapabilitySet};

#[derive(Deserialize)]
struct Models {
    data: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    id: String,
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default)]
    downloaded: bool,
}

pub(super) async fn discover(
    base: &str,
    client: &reqwest::Client,
) -> Result<Vec<DiscoveredModel>, reqwest::Error> {
    let models: Models = client
        .get(format!("{base}/v1/models"))
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(models
        .data
        .into_iter()
        .filter(|entry| entry.downloaded)
        .map(|entry| DiscoveredModel {
            id: entry.id,
            capabilities: parse_capabilities(&entry.labels),
            unknown_capabilities: [Capability::Thinking, Capability::Audio].into(),
            context_window: entry.context_length,
        })
        .collect())
}

fn parse_capabilities(labels: &[String]) -> CapabilitySet {
    labels
        .iter()
        .filter_map(|l| match l.as_str() {
            "chat" => Some(Capability::Completion),
            "tool-calling" => Some(Capability::Tools),
            "vision" => Some(Capability::Vision),
            "embeddings" | "embedding" | "reranking" => Some(Capability::Embedding),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const MODELS: &str = include_str!("../../tests/fixtures/lemonade_models.json");

    fn json_response(body: &str) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_raw(body.to_owned(), "application/json")
    }

    #[tokio::test]
    async fn maps_labels_to_capabilities_and_skips_undownloaded_models() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(json_response(MODELS))
            .mount(&server)
            .await;
        let models = discover(&server.uri(), &reqwest::Client::new())
            .await
            .unwrap();
        assert_eq!(models.len(), 2, "the undownloaded model is skipped");
        let get = |id: &str| models.iter().find(|m| m.id == id).unwrap().clone();

        let small = get("Qwen3-4B-Instruct-2507-GGUF");
        assert_eq!(
            small.capabilities,
            CapabilitySet::from([Capability::Completion, Capability::Tools])
        );
        assert_eq!(
            small.unknown_capabilities,
            CapabilitySet::from([Capability::Thinking, Capability::Audio])
        );
        assert_eq!(small.context_window, Some(262_144));

        let big = get("Qwen3.5-9B-GGUF");
        assert_eq!(
            big.capabilities,
            CapabilitySet::from([
                Capability::Completion,
                Capability::Tools,
                Capability::Vision
            ])
        );
        assert_eq!(
            big.unknown_capabilities,
            CapabilitySet::from([Capability::Thinking, Capability::Audio])
        );
        assert_eq!(big.context_window, Some(262_144));

        assert!(
            models.iter().all(|m| m.id != "Gemma-3-1B-GGUF"),
            "an undownloaded model must never appear"
        );
    }

    #[tokio::test]
    async fn server_error_is_an_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;
        assert!(
            discover(&server.uri(), &reqwest::Client::new())
                .await
                .is_err()
        );
    }

    #[test]
    fn embeddings_and_reranking_labels_both_map_to_embedding() {
        let raw: Vec<String> = ["embeddings", "reranking", "embedding", "mystery"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            parse_capabilities(&raw),
            CapabilitySet::from([Capability::Embedding])
        );
    }
}
