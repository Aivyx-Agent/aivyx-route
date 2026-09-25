//! Discovery → merge → select through the public API only.
#![cfg(feature = "discovery")]

use aivyx_route::discovery::discover_all;
use aivyx_route::{
    DiscoveryOutcome, EndpointRef, Policy, Requirements, RoutingConfig, TaskKind, merge, select,
};
use serde::Deserialize;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Deserialize)]
struct Doc {
    routing: RoutingConfig,
}

fn json_response(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body.to_owned(), "application/json")
}

#[tokio::test]
async fn routes_by_capability_and_task_against_a_mock_ollama() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(json_response(include_str!("fixtures/ollama_tags.json")))
        .mount(&server)
        .await;
    for (model, body) in [
        ("llava:13b", include_str!("fixtures/ollama_show_llava.json")),
        (
            "qwen3-coder:30b",
            include_str!("fixtures/ollama_show_qwen.json"),
        ),
        (
            "nomic-embed-text:latest",
            include_str!("fixtures/ollama_show_embed.json"),
        ),
    ] {
        Mock::given(method("POST"))
            .and(path("/api/show"))
            .and(body_partial_json(json!({ "model": model })))
            .respond_with(json_response(body))
            .mount(&server)
            .await;
    }

    let toml_src = format!(
        r#"
[routing]
enabled = true

[routing.endpoints.gpu]
kind = "ollama"
base_url = "{uri}"

[routing.endpoints.anthropic]
kind = "anthropic"

[[routing.models]]
id = "qwen3-coder:30b"
endpoint = "gpu"
tier = "large"
strengths = ["code", "reasoning"]

[[routing.models]]
id = "llava:13b"
endpoint = "gpu"
tier = "small"

[[routing.models]]
id = "claude-sonnet-5"
endpoint = "anthropic"
tier = "large"
strengths = ["code", "reasoning"]
capabilities = ["completion", "tools", "vision"]
context_window = 200000
"#,
        uri = server.uri()
    );
    let config = toml::from_str::<Doc>(&toml_src).unwrap().routing;

    let reports = discover_all(&config, &reqwest::Client::new()).await;
    assert!(matches!(reports[0].outcome, DiscoveryOutcome::NotProbed)); // "anthropic" sorts first
    assert!(matches!(reports[1].outcome, DiscoveryOutcome::Reached(_)));

    let profiles = merge(&config, &EndpointRef::new("gpu"), &reports);
    assert_eq!(profiles.len(), 4);

    let local = Policy::default();

    // Coding with tools → the large local coder.
    let req = Requirements::builder()
        .task(&TaskKind::CodeEdit, &config.tasks)
        .tools()
        .build();
    assert_eq!(
        select(&req, &profiles, &local).unwrap().model.id,
        "qwen3-coder:30b"
    );

    // An image → the only local vision model, despite its small tier.
    let req = Requirements::builder()
        .task(&TaskKind::Chat, &config.tasks)
        .vision()
        .build();
    assert_eq!(
        select(&req, &profiles, &local).unwrap().model.id,
        "llava:13b"
    );

    // Embeddings → the embedding-only model.
    let req = Requirements::builder()
        .task(&TaskKind::Embed, &config.tasks)
        .build();
    assert_eq!(
        select(&req, &profiles, &local).unwrap().model.id,
        "nomic-embed-text:latest"
    );

    // Huge context: nothing local qualifies; the cloud model does once allowed.
    let req = Requirements::builder()
        .task(&TaskKind::Plan, &config.tasks)
        .min_context(300_000)
        .build();
    let err = select(&req, &profiles, &local).unwrap_err();
    assert!(err.to_string().contains("300000"), "{err}");
    let req = Requirements::builder()
        .task(&TaskKind::Plan, &config.tasks)
        .min_context(150_000)
        .vision()
        .build();
    assert!(select(&req, &profiles, &local).is_err());
    let cloud = Policy {
        allow_cloud: true,
        ..Policy::default()
    };
    assert_eq!(
        select(&req, &profiles, &cloud).unwrap().model.id,
        "claude-sonnet-5"
    );
}
