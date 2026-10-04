//! Ollama: `GET /api/tags` for the model list, then `POST /api/show` per
//! model for `capabilities` and `<arch>.context_length`.

use serde::Deserialize;
use serde_json::{Map, Value, json};

use futures_util::{StreamExt, stream};
use tokio::time::Instant;

use super::{FetchError, SHOW_CONCURRENCY, fetch_json};
use crate::merge::DiscoveredModel;
use crate::profile::{Capability, CapabilitySet};

#[derive(Deserialize)]
struct Tags {
    models: Vec<TagEntry>,
}

#[derive(Deserialize)]
struct TagEntry {
    name: String,
}

#[derive(Deserialize)]
struct Show {
    #[serde(default)]
    capabilities: Vec<String>,
    #[serde(default)]
    model_info: Map<String, Value>,
}

pub(super) async fn discover(
    base: &str,
    client: &reqwest::Client,
    deadline: Instant,
) -> Result<Vec<DiscoveredModel>, FetchError> {
    let tags: Tags = fetch_json(client.get(format!("{base}/api/tags")), deadline).await?;
    let models = stream::iter(tags.models)
        .map(|entry| async move {
            // A failed (or too late) /api/show still leaves a usable model
            // whose capabilities are all unknown.
            match show(base, &entry.name, client, deadline).await {
                Ok(show) => DiscoveredModel {
                    capabilities: parse_capabilities(&show.capabilities),
                    unknown_capabilities: CapabilitySet::new(),
                    context_window: context_length(&show.model_info),
                    id: entry.name,
                },
                Err(_) => DiscoveredModel {
                    id: entry.name,
                    capabilities: CapabilitySet::new(),
                    unknown_capabilities: Capability::ALL.into_iter().collect(),
                    context_window: None,
                },
            }
        })
        .buffered(SHOW_CONCURRENCY)
        .collect()
        .await;
    Ok(models)
}

async fn show(
    base: &str,
    model: &str,
    client: &reqwest::Client,
    deadline: Instant,
) -> Result<Show, FetchError> {
    fetch_json(
        client
            .post(format!("{base}/api/show"))
            .json(&json!({ "model": model })),
        deadline,
    )
    .await
}

/// Ollama's `"image"` capability means image *generation*, not image input,
/// so it is deliberately not mapped to `Vision`.
fn parse_capabilities(raw: &[String]) -> CapabilitySet {
    raw.iter()
        .filter_map(|c| match c.as_str() {
            "completion" => Some(Capability::Completion),
            "tools" => Some(Capability::Tools),
            "vision" => Some(Capability::Vision),
            "thinking" => Some(Capability::Thinking),
            "audio" => Some(Capability::Audio),
            "embedding" => Some(Capability::Embedding),
            _ => None,
        })
        .collect()
}

fn context_length(info: &Map<String, Value>) -> Option<u32> {
    info.iter()
        .find(|(key, _)| key.ends_with(".context_length"))
        .and_then(|(_, v)| v.as_u64())
        .and_then(|n| u32::try_from(n).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const TAGS: &str = include_str!("../../tests/fixtures/ollama_tags.json");
    const SHOW_LLAVA: &str = include_str!("../../tests/fixtures/ollama_show_llava.json");
    const SHOW_QWEN: &str = include_str!("../../tests/fixtures/ollama_show_qwen.json");
    const SHOW_EMBED: &str = include_str!("../../tests/fixtures/ollama_show_embed.json");

    fn json_response(body: &str) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_raw(body.to_owned(), "application/json")
    }

    async fn mount_show(server: &MockServer, model: &str, body: &str) {
        Mock::given(method("POST"))
            .and(path("/api/show"))
            .and(body_partial_json(json!({ "model": model })))
            .respond_with(json_response(body))
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn reads_capabilities_and_context_length_per_model() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(json_response(TAGS))
            .mount(&server)
            .await;
        mount_show(&server, "llava:13b", SHOW_LLAVA).await;
        mount_show(&server, "qwen3-coder:30b", SHOW_QWEN).await;
        mount_show(&server, "nomic-embed-text:latest", SHOW_EMBED).await;

        let models = discover(
            &server.uri(),
            &reqwest::Client::new(),
            crate::discovery::far_deadline(),
        )
        .await
        .unwrap();
        let get = |id: &str| models.iter().find(|m| m.id == id).unwrap().clone();
        assert_eq!(models.len(), 3);
        assert_eq!(
            get("llava:13b").capabilities,
            CapabilitySet::from([Capability::Completion, Capability::Vision])
        );
        assert_eq!(get("llava:13b").context_window, Some(4_096));
        assert_eq!(
            get("qwen3-coder:30b").capabilities,
            CapabilitySet::from([Capability::Completion, Capability::Tools])
        );
        assert_eq!(get("qwen3-coder:30b").context_window, Some(262_144));
        assert_eq!(
            get("nomic-embed-text:latest").capabilities,
            CapabilitySet::from([Capability::Embedding])
        );
        assert!(
            models.iter().all(|m| m.unknown_capabilities.is_empty()),
            "a successful /api/show is authoritative"
        );
    }

    #[tokio::test]
    async fn a_failed_show_keeps_the_model_with_no_capabilities() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(json_response(r#"{"models":[{"name":"mystery:1b"}]}"#))
            .mount(&server)
            .await;
        // No /api/show mock: wiremock answers 404.
        let models = discover(
            &server.uri(),
            &reqwest::Client::new(),
            crate::discovery::far_deadline(),
        )
        .await
        .unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "mystery:1b");
        assert!(models[0].capabilities.is_empty());
        assert_eq!(
            models[0].unknown_capabilities,
            Capability::ALL.into_iter().collect::<CapabilitySet>()
        );
        assert_eq!(models[0].context_window, None);
    }

    #[tokio::test]
    async fn tags_failure_is_an_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(500))
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
    fn image_generation_is_not_vision_and_unknown_strings_are_ignored() {
        let raw: Vec<String> = [
            "completion",
            "image",
            "insert",
            "tools",
            "audio",
            "thinking",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(
            parse_capabilities(&raw),
            CapabilitySet::from([
                Capability::Completion,
                Capability::Tools,
                Capability::Audio,
                Capability::Thinking
            ])
        );
    }
}
