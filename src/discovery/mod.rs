//! Backend discovery (feature `discovery`): ask each configured endpoint
//! which models it serves and what they can do. Runs at startup and on
//! explicit refresh only — never per request. Never fails the caller:
//! problems become `DiscoveryOutcome::Unreachable`.
//!
//! Endpoints are probed concurrently, Ollama's per-model `/api/show` calls
//! [`SHOW_CONCURRENCY`] at a time, and the whole run is bounded by a
//! deadline ([`DISCOVERY_DEADLINE`] unless a `*_within` variant sets one).
//! At the deadline whatever was found is returned: a model whose details
//! didn't arrive keeps every capability unknown, and an endpoint that
//! hadn't listed its models is `Unreachable` ("timed out").

mod lemonade;
mod llama_router;
mod ollama;
mod openai_compat;
pub mod residency;

use std::fmt;
use std::time::Duration;

use futures_util::future::join_all;
use serde::de::DeserializeOwned;
use tokio::time::{Instant, timeout_at};

/// The `reqwest` this crate is built against. Consumers on a different
/// reqwest major build the client as
/// `aivyx_route::discovery::reqwest::Client::new()`.
pub use reqwest;

use crate::config::{EndpointConfig, EndpointKind, RoutingConfig};
use crate::merge::{DiscoveryOutcome, DiscoveryReport};
use crate::profile::EndpointRef;

/// Timeout for each individual HTTP request made during discovery.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// The default bound on a whole [`discover`] / [`discover_all`] run.
pub const DISCOVERY_DEADLINE: Duration = Duration::from_secs(15);

/// How many per-model requests (Ollama `/api/show`) one endpoint has in
/// flight at once.
pub const SHOW_CONCURRENCY: usize = 8;

/// How long past the deadline a probe may take to hand back what it found
/// before it is abandoned outright.
const DEADLINE_GRACE: Duration = Duration::from_millis(250);

/// The largest discovery or residency response body read; anything
/// larger is refused.
pub const MAX_BODY_BYTES: u64 = 8 << 20;

/// A client for discovery and residency that never follows redirects.
/// [`discover`] and [`residency::collect`] refuse a redirected response
/// with any client, but a client that follows one has already sent the
/// second request.
pub fn client() -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
}

/// Why one discovery request failed.
#[derive(Debug)]
pub(crate) enum FetchError {
    Http(reqwest::Error),
    /// The overall discovery deadline passed.
    TimedOut,
    /// The server answered with (or the client followed) a redirect.
    Redirected,
    /// The body exceeded [`MAX_BODY_BYTES`].
    TooLarge,
    Json(serde_json::Error),
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FetchError::Http(e) => e.fmt(f),
            FetchError::TimedOut => f.write_str("timed out: the discovery deadline passed"),
            FetchError::Redirected => {
                f.write_str("the server answered with a redirect, which is not followed")
            }
            FetchError::TooLarge => write!(f, "response too large (over {MAX_BODY_BYTES} bytes)"),
            FetchError::Json(e) => write!(f, "unexpected response: {e}"),
        }
    }
}

impl From<reqwest::Error> for FetchError {
    fn from(e: reqwest::Error) -> Self {
        FetchError::Http(e)
    }
}

/// Send `request` and parse its JSON body, within [`REQUEST_TIMEOUT`] and
/// before `deadline`. Redirects and bodies over [`MAX_BODY_BYTES`] are
/// refused.
pub(crate) async fn fetch_json<T: DeserializeOwned>(
    request: reqwest::RequestBuilder,
    deadline: Instant,
) -> Result<T, FetchError> {
    let fetch = async {
        let (client, request) = request.timeout(REQUEST_TIMEOUT).build_split();
        let request = request?;
        let url = request.url().clone();
        let mut response = client.execute(request).await?;
        if response.status().is_redirection() || *response.url() != url {
            return Err(FetchError::Redirected);
        }
        response = response.error_for_status()?;
        if response
            .content_length()
            .is_some_and(|n| n > MAX_BODY_BYTES)
        {
            return Err(FetchError::TooLarge);
        }
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            if (body.len() + chunk.len()) as u64 > MAX_BODY_BYTES {
                return Err(FetchError::TooLarge);
            }
            body.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&body).map_err(FetchError::Json)
    };
    timeout_at(deadline, fetch)
        .await
        .unwrap_or(Err(FetchError::TimedOut))
}

/// Probe one endpoint, within [`DISCOVERY_DEADLINE`].
pub async fn discover(
    endpoint: &EndpointRef,
    config: &EndpointConfig,
    client: &reqwest::Client,
) -> DiscoveryReport {
    discover_within(endpoint, config, client, DISCOVERY_DEADLINE).await
}

/// Probe one endpoint, within `deadline`.
pub async fn discover_within(
    endpoint: &EndpointRef,
    config: &EndpointConfig,
    client: &reqwest::Client,
    deadline: Duration,
) -> DiscoveryReport {
    discover_until(endpoint, config, client, Instant::now() + deadline).await
}

async fn discover_until(
    endpoint: &EndpointRef,
    config: &EndpointConfig,
    client: &reqwest::Client,
    deadline: Instant,
) -> DiscoveryReport {
    // Every request already stops at `deadline`; this is the backstop.
    let outcome = timeout_at(
        deadline + DEADLINE_GRACE,
        probe(endpoint, config, client, deadline),
    )
    .await
    .unwrap_or_else(|_| DiscoveryOutcome::Unreachable(FetchError::TimedOut.to_string()));
    DiscoveryReport {
        endpoint: endpoint.clone(),
        outcome,
    }
}

async fn probe(
    endpoint: &EndpointRef,
    config: &EndpointConfig,
    client: &reqwest::Client,
    deadline: Instant,
) -> DiscoveryOutcome {
    match config.kind {
        EndpointKind::Anthropic | EndpointKind::Openai => DiscoveryOutcome::NotProbed,
        kind => match config.base_url() {
            None => DiscoveryOutcome::Unreachable(format!(
                "no base_url configured for endpoint `{endpoint}`"
            )),
            Some(base) => {
                let base = base.trim_end_matches('/');
                let result = match kind {
                    EndpointKind::Ollama => ollama::discover(base, client, deadline).await,
                    EndpointKind::LlamaRouter => {
                        llama_router::discover(base, client, deadline).await
                    }
                    EndpointKind::Lemonade => lemonade::discover(base, client, deadline).await,
                    _ => openai_compat::discover(base, client, deadline).await,
                };
                match result {
                    Ok(models) => DiscoveryOutcome::Reached(models),
                    Err(e) => DiscoveryOutcome::Unreachable(e.to_string()),
                }
            }
        },
    }
}

/// Probe every `[routing.endpoints]` entry concurrently, within
/// [`DISCOVERY_DEADLINE`]. Reports are in name order.
pub async fn discover_all(
    config: &RoutingConfig,
    client: &reqwest::Client,
) -> Vec<DiscoveryReport> {
    discover_all_within(config, client, DISCOVERY_DEADLINE).await
}

/// [`discover_all`] with its own overall deadline.
pub async fn discover_all_within(
    config: &RoutingConfig,
    client: &reqwest::Client,
    deadline: Duration,
) -> Vec<DiscoveryReport> {
    let deadline = Instant::now() + deadline;
    let names: Vec<EndpointRef> = config
        .endpoints
        .keys()
        .map(|name| EndpointRef::new(name.clone()))
        .collect();
    join_all(
        names
            .iter()
            .zip(config.endpoints.values())
            .map(|(name, endpoint)| discover_until(name, endpoint, client, deadline)),
    )
    .await
}

/// A deadline no test reaches.
#[cfg(test)]
pub(crate) fn far_deadline() -> Instant {
    Instant::now() + Duration::from_secs(60)
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

    const SHOW_QWEN: &str = include_str!("../../tests/fixtures/ollama_show_qwen.json");

    fn json(body: &str) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_raw(body.to_owned(), "application/json")
    }

    /// An Ollama server with `n` models whose `/api/tags` takes `tags_delay`
    /// and every `/api/show` `show_delay`.
    async fn slow_ollama(n: usize, tags_delay: Duration, show_delay: Duration) -> MockServer {
        let server = MockServer::start().await;
        let names: Vec<String> = (0..n).map(|i| format!(r#"{{"name":"m{i}:8b"}}"#)).collect();
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(
                json(&format!(r#"{{"models":[{}]}}"#, names.join(","))).set_delay(tags_delay),
            )
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/show"))
            .respond_with(json(SHOW_QWEN).set_delay(show_delay))
            .mount(&server)
            .await;
        server
    }

    fn models_of(outcome: &DiscoveryOutcome) -> &[crate::merge::DiscoveredModel] {
        match outcome {
            DiscoveryOutcome::Reached(models) => models,
            other => panic!("expected Reached, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn show_calls_run_concurrently() {
        let server = slow_ollama(20, Duration::ZERO, Duration::from_secs(1)).await;
        let started = std::time::Instant::now();
        let r = discover(
            &EndpointRef::new("ollama"),
            &cfg(EndpointKind::Ollama, Some(&server.uri())),
            &reqwest::Client::new(),
        )
        .await;
        let elapsed = started.elapsed();
        let models = models_of(&r.outcome);
        assert_eq!(models.len(), 20);
        assert!(
            models.iter().all(|m| m.unknown_capabilities.is_empty()),
            "every /api/show answered"
        );
        // Serially this is 20 s; 8 at a time, 3 rounds.
        assert!(elapsed < Duration::from_secs(6), "took {elapsed:?}");
    }

    #[tokio::test]
    async fn endpoints_are_probed_concurrently() {
        let mut config = RoutingConfig::default();
        let mut servers = Vec::new();
        for name in ["a", "b", "c"] {
            let server = slow_ollama(1, Duration::from_secs(1), Duration::ZERO).await;
            config
                .endpoints
                .insert(name.into(), cfg(EndpointKind::Ollama, Some(&server.uri())));
            servers.push(server);
        }
        let started = std::time::Instant::now();
        let reports = discover_all(&config, &reqwest::Client::new()).await;
        let elapsed = started.elapsed();
        let names: Vec<&str> = reports.iter().map(|r| r.endpoint.as_str()).collect();
        assert_eq!(names, ["a", "b", "c"]);
        assert!(reports.iter().all(|r| models_of(&r.outcome).len() == 1));
        assert!(elapsed < Duration::from_millis(2_500), "took {elapsed:?}");
    }

    /// A wedged server (each request just under the per-request timeout)
    /// can't hold discovery past its deadline: what was found is returned,
    /// the rest is reported as timed out.
    #[tokio::test]
    async fn a_wedged_server_cannot_exceed_the_deadline() {
        let wedged_tags = slow_ollama(1, Duration::from_secs(4), Duration::ZERO).await;
        let wedged_shows = slow_ollama(20, Duration::ZERO, Duration::from_secs(4)).await;
        let fast = slow_ollama(2, Duration::ZERO, Duration::ZERO).await;
        let mut config = RoutingConfig::default();
        for (name, server) in [
            ("a-wedged-tags", &wedged_tags),
            ("b-wedged-shows", &wedged_shows),
            ("c-fast", &fast),
        ] {
            config
                .endpoints
                .insert(name.into(), cfg(EndpointKind::Ollama, Some(&server.uri())));
        }
        let deadline = Duration::from_millis(1_500);
        let started = std::time::Instant::now();
        let reports = discover_all_within(&config, &reqwest::Client::new(), deadline).await;
        let elapsed = started.elapsed();
        assert!(
            elapsed < deadline + Duration::from_millis(750),
            "took {elapsed:?}"
        );
        match &reports[0].outcome {
            DiscoveryOutcome::Unreachable(why) => assert!(why.contains("timed out"), "{why}"),
            other => panic!("expected Unreachable, got {other:?}"),
        }
        // The model list arrived; its capabilities did not.
        let models = models_of(&reports[1].outcome);
        assert_eq!(models.len(), 20);
        assert!(
            models
                .iter()
                .all(|m| m.unknown_capabilities.len() == crate::profile::Capability::ALL.len())
        );
        let models = models_of(&reports[2].outcome);
        assert!(models.iter().all(|m| m.unknown_capabilities.is_empty()));
    }

    #[tokio::test]
    async fn one_endpoint_honours_a_deadline_too() {
        let wedged = slow_ollama(1, Duration::from_secs(4), Duration::ZERO).await;
        let started = std::time::Instant::now();
        let r = discover_within(
            &EndpointRef::new("w"),
            &cfg(EndpointKind::Ollama, Some(&wedged.uri())),
            &reqwest::Client::new(),
            Duration::from_millis(500),
        )
        .await;
        assert!(started.elapsed() < Duration::from_millis(1_250));
        match r.outcome {
            DiscoveryOutcome::Unreachable(why) => assert!(why.contains("timed out"), "{why}"),
            other => panic!("expected Unreachable, got {other:?}"),
        }
    }

    fn unreachable_reason(outcome: DiscoveryOutcome) -> String {
        match outcome {
            DiscoveryOutcome::Unreachable(why) => why,
            other => panic!("expected Unreachable, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn an_oversized_response_is_refused() {
        let server = MockServer::start().await;
        let padding = "x".repeat((MAX_BODY_BYTES + 1) as usize);
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(json(&format!(r#"{{"data":[{{"id":"{padding}"}}]}}"#)))
            .mount(&server)
            .await;
        let r = discover(
            &EndpointRef::new("big"),
            &cfg(EndpointKind::OpenaiCompat, Some(&server.uri())),
            &reqwest::Client::new(),
        )
        .await;
        let why = unreachable_reason(r.outcome);
        assert!(why.contains("too large"), "{why}");
    }

    #[tokio::test]
    async fn redirects_are_not_followed() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(302).insert_header("location", "/elsewhere"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/elsewhere"))
            .respond_with(json(include_str!(
                "../../tests/fixtures/openai_models.json"
            )))
            .mount(&server)
            .await;
        let config = cfg(EndpointKind::OpenaiCompat, Some(&server.uri()));
        // A consumer's own (redirect-following) client, and ours.
        for client in [reqwest::Client::new(), client().unwrap()] {
            let r = discover(&EndpointRef::new("moved"), &config, &client).await;
            let why = unreachable_reason(r.outcome);
            assert!(why.contains("redirect"), "{why}");
        }
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
