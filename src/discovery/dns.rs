//! A connect-time DNS resolver that refuses non-local answers.

use reqwest::dns::{Addrs, Name, Resolve, Resolving};

use std::sync::Arc;

use crate::guard::{Resolve as BlockingResolve, SystemResolver, resolve_local_with};

/// A reqwest DNS resolver that resolves through the system and fails unless
/// every address is local ([`resolve_local`]); the `NotLocal` error is the
/// boxed error, so reqwest's error chain carries its text.
///
/// It is built on this crate's own blocking [`Resolve`](crate::Resolve) seam
/// (not `reqwest::dns::Resolve`, which `LocalOnlyDns` itself implements);
/// [`with_resolver`](Self::with_resolver) swaps the system resolver for another.
///
/// Intended use:
///
/// ```ignore
/// let client = reqwest::Client::builder()
///     .dns_resolver(std::sync::Arc::new(aivyx_route::discovery::LocalOnlyDns::new()))
///     .redirect(reqwest::redirect::Policy::none())
///     .no_proxy()
///     .build()?;
/// ```
///
/// `.no_proxy()` matters: reqwest otherwise sends requests to
/// `HTTP_PROXY`/`ALL_PROXY`, and a proxy given as an IP literal bypasses
/// both this resolver and the URL check.
///
/// Names that are local by spelling (`.lan`, `.local`, `.internal`,
/// `.home.arpa`, single-label names, `localhost`) pass `require_local`
/// without a lookup, so the per-request check does not protect them; only
/// this connect-time resolver does (e.g. a single-label `gpu` expanded by a
/// search domain to a public host).
///
/// With such a client, connections to a host name can only ever go to local
/// addresses, whatever DNS answers between an earlier `require_local` check
/// and the connection. reqwest does **not** call the resolver for IP-literal
/// URLs, so keep checking the URL with [`require_local`](crate::require_local)
/// as well: use both. [`client()`](super::client) already installs it.
#[derive(Clone)]
pub struct LocalOnlyDns {
    resolver: Arc<dyn BlockingResolve>,
}

impl LocalOnlyDns {
    /// Resolves through the operating system.
    pub fn new() -> Self {
        Self::with_resolver(Arc::new(SystemResolver))
    }

    /// Resolves through `resolver` (this crate's blocking
    /// [`Resolve`](crate::Resolve)), still refusing any non-local answer.
    pub fn with_resolver(resolver: Arc<dyn BlockingResolve>) -> Self {
        Self { resolver }
    }
}

impl Default for LocalOnlyDns {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LocalOnlyDns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LocalOnlyDns")
    }
}

impl Resolve for LocalOnlyDns {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_owned();
        let resolver = Arc::clone(&self.resolver);
        Box::pin(async move {
            // Port 0: reqwest substitutes the URL's port.
            let addrs = tokio::task::spawn_blocking(move || {
                resolve_local_with(&host, 0, resolver.as_ref())
            })
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
            Ok(Box::new(addrs.into_iter()) as Addrs)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn refuses_a_public_result_and_names_why() {
        // An IP literal needs no lookup, so this is offline.
        let name: Name = "203.0.113.7".parse().unwrap();
        let err = LocalOnlyDns::new()
            .resolve(name)
            .await
            .err()
            .expect("refused");
        let text = err.to_string();
        assert!(
            text.contains("is not local") && text.contains("public address 203.0.113.7"),
            "{text}"
        );
        let name: Name = "10.1.2.3".parse().unwrap();
        let addrs: Vec<_> = LocalOnlyDns::new()
            .resolve(name)
            .await
            .ok()
            .unwrap()
            .collect();
        assert_eq!(addrs.len(), 1);
    }

    struct Public;

    impl BlockingResolve for Public {
        fn resolve(&self, _host: &str, _port: u16) -> std::io::Result<Vec<std::net::IpAddr>> {
            Ok(vec!["1.2.3.4".parse().unwrap()])
        }
    }

    #[tokio::test]
    async fn refuses_a_name_that_resolves_publicly() {
        let dns = LocalOnlyDns::with_resolver(Arc::new(Public));
        let name: Name = "ai.example.com".parse().unwrap();
        let text = dns.resolve(name).await.err().expect("refused").to_string();
        assert!(
            text.contains("resolves to a public address 1.2.3.4"),
            "{text}"
        );
        assert!(text.contains("ai.example.com is not local"), "{text}");
    }
}
