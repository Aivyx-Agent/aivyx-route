//! A connect-time DNS resolver that refuses non-local answers.

use reqwest::dns::{Addrs, Name, Resolve, Resolving};

use crate::guard::resolve_local;

/// A reqwest DNS resolver that resolves through the system and fails unless
/// every address is local ([`resolve_local`]); the `NotLocal` error is the
/// boxed error, so reqwest's error chain carries its text.
///
/// Intended use:
///
/// ```ignore
/// let client = reqwest::Client::builder()
///     .dns_resolver(std::sync::Arc::new(aivyx_route::discovery::LocalOnlyDns))
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
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalOnlyDns;

impl Resolve for LocalOnlyDns {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_owned();
        Box::pin(async move {
            // Port 0: reqwest substitutes the URL's port.
            let addrs = tokio::task::spawn_blocking(move || resolve_local(&host, 0))
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
        let err = LocalOnlyDns.resolve(name).await.err().expect("refused");
        let text = err.to_string();
        assert!(
            text.contains("is not local") && text.contains("public address 203.0.113.7"),
            "{text}"
        );
        let name: Name = "10.1.2.3".parse().unwrap();
        let addrs: Vec<_> = LocalOnlyDns.resolve(name).await.ok().unwrap().collect();
        assert_eq!(addrs.len(), 1);
    }
}
