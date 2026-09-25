//! llama-server router mode: `GET /models` lists every model with its
//! `architecture.input_modalities`.

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
    architecture: Option<Architecture>,
}

#[derive(Deserialize)]
struct Architecture {
    #[serde(default)]
    input_modalities: Vec<String>,
}

pub(super) async fn discover(
    base: &str,
    client: &reqwest::Client,
) -> Result<Vec<DiscoveredModel>, reqwest::Error> {
    let models: Models = client
        .get(format!("{base}/models"))
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(models
        .data
        .into_iter()
        .map(|entry| {
            let mut capabilities = CapabilitySet::new();
            // Modalities make vision and audio known; nothing reports the
            // rest. Without an architecture block nothing is known at all.
            let unknown_capabilities: CapabilitySet = match &entry.architecture {
                Some(_) => [
                    Capability::Completion,
                    Capability::Tools,
                    Capability::Thinking,
                    Capability::Embedding,
                ]
                .into(),
                None => Capability::ALL.into(),
            };
            let modalities = entry
                .architecture
                .map(|a| a.input_modalities)
                .unwrap_or_default();
            for modality in &modalities {
                match modality.as_str() {
                    "image" => {
                        capabilities.insert(Capability::Vision);
                    }
                    "audio" => {
                        capabilities.insert(Capability::Audio);
                    }
                    _ => {}
                }
            }
            DiscoveredModel {
                id: entry.id,
                capabilities,
                unknown_capabilities,
                context_window: None,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn maps_input_modalities_to_capabilities() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                include_str!("../../tests/fixtures/llama_router_models.json"),
                "application/json",
            ))
            .mount(&server)
            .await;
        let models = discover(&server.uri(), &reqwest::Client::new())
            .await
            .unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M");
        assert_eq!(
            models[0].capabilities,
            CapabilitySet::from([Capability::Vision])
        );
        assert_eq!(
            models[0].unknown_capabilities,
            CapabilitySet::from([
                Capability::Completion,
                Capability::Tools,
                Capability::Thinking,
                Capability::Embedding
            ]),
            "modalities make vision and audio known"
        );
        assert_eq!(models[1].id, "unsloth/Qwen3-8B-GGUF:Q4_K_M");
        assert!(
            models[1].capabilities.is_empty(),
            "no architecture block ⇒ no capabilities"
        );
        assert_eq!(
            models[1].unknown_capabilities,
            Capability::ALL.into_iter().collect::<CapabilitySet>(),
            "no architecture block ⇒ not even modalities are known"
        );
        assert!(models.iter().all(|m| m.context_window.is_none()));
    }

    #[tokio::test]
    async fn server_error_is_an_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;
        assert!(
            discover(&server.uri(), &reqwest::Client::new())
                .await
                .is_err()
        );
    }
}
