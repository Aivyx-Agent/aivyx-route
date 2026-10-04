//! Lemonade Server: `GET /v1/models` lists the catalog (downloaded models
//! only); its `labels` map onto capabilities. A `reasoning` label means the
//! model thinks, but not every thinking model carries it, so without it
//! thinking stays unknown. Audio is never labelled and stays unknown.

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
    deadline: Instant,
) -> Result<Vec<DiscoveredModel>, FetchError> {
    let models: Models = fetch_json(client.get(api_url(base, "models")), deadline).await?;
    Ok(models
        .data
        .into_iter()
        .filter(|entry| entry.downloaded)
        .map(|entry| {
            let (capabilities, unknown_capabilities) = capabilities(&entry.labels);
            DiscoveredModel {
                id: entry.id,
                capabilities,
                unknown_capabilities,
                context_window: entry.context_length,
            }
        })
        .collect())
}

/// `{base}/v1/{path}` for a Lemonade base given as `.../api` (the
/// documented form) or, tolerated, `.../api/v1`.
pub(super) fn api_url(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    let base = base.strip_suffix("/v1").unwrap_or(base);
    format!("{base}/v1/{path}")
}

/// Known-present and unknown capabilities from Lemonade's labels.
fn capabilities(labels: &[String]) -> (CapabilitySet, CapabilitySet) {
    let mut present = parse_capabilities(labels);
    let mut unknown = CapabilitySet::from([Capability::Audio]);
    if labels.iter().any(|l| l == "reasoning") {
        present.insert(Capability::Thinking);
    } else {
        unknown.insert(Capability::Thinking);
    }
    (present, unknown)
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
        let models = discover(
            &server.uri(),
            &reqwest::Client::new(),
            crate::discovery::far_deadline(),
        )
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
            discover(
                &server.uri(),
                &reqwest::Client::new(),
                crate::discovery::far_deadline()
            )
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

    #[test]
    fn a_reasoning_label_marks_the_model_as_thinking() {
        let raw = vec!["chat".to_string(), "reasoning".to_string()];
        let (caps, unknown) = capabilities(&raw);
        assert_eq!(
            caps,
            CapabilitySet::from([Capability::Completion, Capability::Thinking])
        );
        assert_eq!(unknown, CapabilitySet::from([Capability::Audio]));
    }

    #[test]
    fn without_a_reasoning_label_thinking_stays_unknown() {
        // Qwen3.5-9B thinks but carries no `reasoning` label.
        let raw = vec!["chat".to_string()];
        let (caps, unknown) = capabilities(&raw);
        assert_eq!(caps, CapabilitySet::from([Capability::Completion]));
        assert_eq!(
            unknown,
            CapabilitySet::from([Capability::Thinking, Capability::Audio])
        );
    }

    #[test]
    fn api_url_accepts_bases_with_or_without_v1() {
        for base in [
            "http://h:13305/api",
            "http://h:13305/api/",
            "http://h:13305/api/v1",
            "http://h:13305/api/v1/",
        ] {
            assert_eq!(
                api_url(base, "models"),
                "http://h:13305/api/v1/models",
                "{base}"
            );
        }
    }

    #[tokio::test]
    async fn discovery_works_on_a_base_ending_in_v1() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(json_response(MODELS))
            .mount(&server)
            .await;
        let models = discover(
            &format!("{}/api/v1", server.uri()),
            &reqwest::Client::new(),
            crate::discovery::far_deadline(),
        )
        .await
        .unwrap();
        assert_eq!(models.len(), 2);
    }
}
