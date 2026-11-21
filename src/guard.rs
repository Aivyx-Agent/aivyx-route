//! The local-only guard: `require_local` accepts a URL only when it points
//! at a server on this machine or the operator's own network.
//!
//! Local means `url_locality(url) == Local`, or a domain that resolves to at
//! least one address, every one of which is local under the same IP rules.

use std::fmt;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};

use url::{Host, Url};

use crate::locality::{domain_is_local, ip_is_local, parse_http_url};

/// Resolves a host to its addresses (a seam for tests).
pub trait Resolve: Send + Sync {
    fn resolve(&self, host: &str, port: u16) -> std::io::Result<Vec<IpAddr>>;
}

/// Resolves through the operating system (`std::net::ToSocketAddrs`).
pub struct SystemResolver;

impl Resolve for SystemResolver {
    fn resolve(&self, host: &str, port: u16) -> std::io::Result<Vec<IpAddr>> {
        Ok((host, port)
            .to_socket_addrs()?
            .map(|addr| addr.ip())
            .collect())
    }
}

/// A URL that is not local, with the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotLocal {
    /// The URL with any credentials removed.
    pub url: String,
    pub reason: NotLocalReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotLocalReason {
    CloudHost,
    PublicAddress(IpAddr),
    DoesNotResolve,
    NotHttp,
}

impl fmt::Display for NotLocal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} is not local (", self.url)?;
        match &self.reason {
            NotLocalReason::CloudHost => write!(f, "cloud host")?,
            NotLocalReason::PublicAddress(ip) => {
                write!(f, "resolves to a public address {ip}")?;
            }
            NotLocalReason::DoesNotResolve => write!(f, "doesn't resolve")?,
            NotLocalReason::NotHttp => write!(f, "not an http(s) URL")?,
        }
        write!(
            f,
            ") — aivyx is local-only; point it at a model server on this machine or your own network"
        )
    }
}

impl std::error::Error for NotLocal {}

/// Known cloud API hosts, reported as such without a lookup.
const CLOUD_HOSTS: &[&str] = &[
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "openrouter.ai",
    "api.groq.com",
    "api.mistral.ai",
    "api.deepseek.com",
    "api.x.ai",
    "api.together.xyz",
    "api.fireworks.ai",
    "api.perplexity.ai",
    "api.cohere.com",
];

fn is_cloud_host(domain: &str) -> bool {
    let domain = domain.strip_suffix('.').unwrap_or(domain);
    CLOUD_HOSTS.contains(&domain) || domain.ends_with(".openai.azure.com")
}

/// The addresses `host:port` connects to, `Ok` only when all of them are
/// local (system resolver). The reqwest-independent core of the connect-time
/// guard ([`LocalOnlyDns`](crate::discovery::LocalOnlyDns)).
pub fn resolve_local(host: &str, port: u16) -> Result<Vec<SocketAddr>, NotLocal> {
    resolve_local_with(host, port, &SystemResolver)
}

/// [`resolve_local`] with an explicit resolver. An IP-literal host (bracketed
/// or not) is returned as-is when local; a known cloud host is refused without
/// a lookup; any other name must resolve to at least one address, all local
/// (even a `.lan`/`.local` name: what it connects to is what counts).
/// `NotLocal.url` is `host:port`.
pub fn resolve_local_with(
    host: &str,
    port: u16,
    r: &dyn Resolve,
) -> Result<Vec<SocketAddr>, NotLocal> {
    let fail = |reason| {
        Err(NotLocal {
            url: format!("{host}:{port}"),
            reason,
        })
    };
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = bare.parse::<IpAddr>() {
        return if ip_is_local(ip) {
            Ok(vec![SocketAddr::new(ip, port)])
        } else {
            fail(NotLocalReason::PublicAddress(ip))
        };
    }
    if is_cloud_host(&host.to_ascii_lowercase()) {
        return fail(NotLocalReason::CloudHost);
    }
    match r.resolve(host, port) {
        Err(_) => fail(NotLocalReason::DoesNotResolve),
        Ok(addrs) if addrs.is_empty() => fail(NotLocalReason::DoesNotResolve),
        Ok(addrs) => match addrs.iter().find(|ip| !ip_is_local(**ip)) {
            Some(ip) => fail(NotLocalReason::PublicAddress(*ip)),
            None => Ok(addrs
                .into_iter()
                .map(|ip| SocketAddr::new(ip, port))
                .collect()),
        },
    }
}

/// `Ok` when `url` is local, resolving domains through the system resolver.
pub fn require_local(url: &str) -> Result<(), NotLocal> {
    require_local_with(url, &SystemResolver)
}

/// [`require_local`] with an explicit resolver.
pub fn require_local_with(url: &str, r: &dyn Resolve) -> Result<(), NotLocal> {
    let fail = |url: String, reason| Err(NotLocal { url, reason });
    let Some(parsed) = parse_http_url(url) else {
        return fail(strip_credentials(url), NotLocalReason::NotHttp);
    };
    let shown = {
        let mut clean = parsed.clone();
        let _ = clean.set_username("");
        let _ = clean.set_password(None);
        clean.to_string()
    };
    match parsed.host() {
        Some(Host::Ipv4(ip)) => check_ip(shown, IpAddr::V4(ip)),
        Some(Host::Ipv6(ip)) => check_ip(shown, IpAddr::V6(ip)),
        Some(Host::Domain(domain)) if !domain.is_empty() => {
            if is_cloud_host(domain) {
                return fail(shown, NotLocalReason::CloudHost);
            }
            if domain_is_local(domain) {
                return Ok(());
            }
            // http(s) always has a known default port, so this never falls back.
            let port = parsed.port_or_known_default().unwrap_or(80);
            match r.resolve(domain, port) {
                Err(_) => fail(shown, NotLocalReason::DoesNotResolve),
                Ok(addrs) if addrs.is_empty() => fail(shown, NotLocalReason::DoesNotResolve),
                Ok(addrs) => match addrs.into_iter().find(|ip| !ip_is_local(*ip)) {
                    Some(ip) => fail(shown, NotLocalReason::PublicAddress(ip)),
                    None => Ok(()),
                },
            }
        }
        _ => fail(shown, NotLocalReason::NotHttp),
    }
}

fn check_ip(shown: String, ip: IpAddr) -> Result<(), NotLocal> {
    if ip_is_local(ip) {
        Ok(())
    } else {
        Err(NotLocal {
            url: shown,
            reason: NotLocalReason::PublicAddress(ip),
        })
    }
}

/// The URL as given, minus any `user:pass@` (also when it doesn't parse).
fn strip_credentials(url: &str) -> String {
    match Url::parse(url.trim()) {
        Ok(mut u) if u.has_host() => {
            let _ = u.set_username("");
            let _ = u.set_password(None);
            u.to_string()
        }
        _ => strip_userinfo_text(url),
    }
}

/// Textual fallback: cut `userinfo@` from the authority, which runs from
/// `://` (or the start) to the first `/`, `?` or `#`, at its last `@`.
fn strip_userinfo_text(url: &str) -> String {
    let start = url.find("://").map_or(0, |i| i + 3);
    let end = url[start..]
        .find(['/', '?', '#'])
        .map_or(url.len(), |i| start + i);
    match url[start..end].rfind('@') {
        Some(at) => format!("{}{}", &url[..start], &url[start + at + 1..]),
        None => url.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct Stub {
        map: HashMap<&'static str, Result<Vec<IpAddr>, ()>>,
        calls: AtomicUsize,
    }

    impl Resolve for Stub {
        fn resolve(&self, host: &str, _port: u16) -> std::io::Result<Vec<IpAddr>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match self.map.get(host) {
                Some(Ok(v)) => Ok(v.clone()),
                _ => Err(std::io::Error::other("nxdomain")),
            }
        }
    }

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    fn reason(url: &str, stub: &Stub) -> NotLocalReason {
        require_local_with(url, stub).unwrap_err().reason
    }

    #[test]
    fn local_ip_ranges_are_ok() {
        let stub = Stub::default();
        for u in [
            "http://127.0.0.1:8090",
            "http://10.1.2.3/",
            "http://172.16.0.1/",
            "http://192.168.1.5/",
            "http://169.254.1.1/",
            "http://100.64.0.1/",
            "http://0.0.0.0/",
            "http://[::1]:8080/",
            "http://[fd00::1]/",
            "http://[fe80::1]/",
            "http://[fc00::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[::]/",
        ] {
            assert!(require_local_with(u, &stub).is_ok(), "{u}");
        }
        assert_eq!(stub.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn public_ip_is_rejected() {
        let stub = Stub::default();
        assert_eq!(
            reason("http://8.8.8.8/", &stub),
            NotLocalReason::PublicAddress(ip("8.8.8.8"))
        );
        assert_eq!(
            reason("http://[2606:4700::1111]/", &stub),
            NotLocalReason::PublicAddress(ip("2606:4700::1111"))
        );
    }

    #[test]
    fn local_names_need_no_lookup() {
        let stub = Stub::default();
        for u in [
            "http://localhost:1/",
            "http://gpu.lan/",
            "http://box.local/",
            "http://x.home.arpa/",
            "http://gpu/",
        ] {
            assert!(require_local_with(u, &stub).is_ok(), "{u}");
        }
        assert_eq!(stub.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn cloud_hosts_are_named_without_lookup() {
        let stub = Stub::default();
        for u in [
            "https://api.openai.com/v1",
            "https://api.anthropic.com",
            "https://acme.openai.azure.com/x",
        ] {
            assert_eq!(reason(u, &stub), NotLocalReason::CloudHost, "{u}");
        }
        assert_eq!(stub.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn domains_resolve_all_local_or_fail() {
        let mut stub = Stub::default();
        stub.map
            .insert("ai.example.com", Ok(vec![ip("192.168.1.5")]));
        assert!(require_local_with("https://ai.example.com/", &stub).is_ok());
        stub.map
            .insert("ai.example.com", Ok(vec![ip("192.168.1.5"), ip("1.2.3.4")]));
        assert_eq!(
            reason("https://ai.example.com/", &stub),
            NotLocalReason::PublicAddress(ip("1.2.3.4"))
        );
    }

    #[test]
    fn resolver_error_or_empty_is_does_not_resolve() {
        let mut stub = Stub::default();
        assert_eq!(
            reason("http://nope.example.com/", &stub),
            NotLocalReason::DoesNotResolve
        );
        stub.map.insert("empty.example.com", Ok(vec![]));
        assert_eq!(
            reason("http://empty.example.com/", &stub),
            NotLocalReason::DoesNotResolve
        );
    }

    #[test]
    fn non_http_is_not_http() {
        let stub = Stub::default();
        for u in ["ftp://host/x", "not a url", "unix:/x"] {
            assert_eq!(reason(u, &stub), NotLocalReason::NotHttp, "{u}");
        }
    }

    #[test]
    fn display_strips_credentials_and_matches_text() {
        let stub = Stub::default();
        let text = require_local_with("http://u:p@1.2.3.4/", &stub)
            .unwrap_err()
            .to_string();
        assert!(!text.contains("u:p"), "{text}");
        assert_eq!(
            text,
            "http://1.2.3.4/ is not local (resolves to a public address 1.2.3.4) — aivyx is local-only; point it at a model server on this machine or your own network"
        );
        let bad = require_local_with("ftp://u:p@host/x", &stub).unwrap_err();
        assert!(!bad.to_string().contains("u:p"));
    }

    #[test]
    fn system_resolver_handles_ip_literals() {
        assert_eq!(
            SystemResolver.resolve("127.0.0.1", 80).unwrap(),
            vec![ip("127.0.0.1")]
        );
        assert!(require_local("http://127.0.0.1:8090").is_ok());
    }

    #[test]
    fn unparsable_urls_do_not_leak_credentials() {
        let stub = Stub::default();
        for u in [
            "http://u:p@host:99999/",
            "http://u:p@[zz]/",
            "u:p@host:99999",
        ] {
            let e = require_local_with(u, &stub).unwrap_err();
            assert_eq!(e.reason, NotLocalReason::NotHttp, "{u}");
            let text = e.to_string();
            assert!(!text.contains("u:p") && !text.contains("p@"), "{text}");
        }
    }

    #[test]
    fn a_leading_slash_path_is_not_a_bare_host() {
        let stub = Stub::default();
        assert_eq!(reason("/v1/chat", &stub), NotLocalReason::NotHttp);
        assert_eq!(reason("\\v1", &stub), NotLocalReason::NotHttp);
        assert_eq!(reason(" /v1", &stub), NotLocalReason::NotHttp);
    }

    #[test]
    fn bare_host_port_and_case() {
        let mut stub = Stub::default();
        assert_eq!(
            reason("api.openai.com:443", &stub),
            NotLocalReason::CloudHost
        );
        assert_eq!(
            reason("HTTP://API.OPENAI.COM/", &stub),
            NotLocalReason::CloudHost
        );
        assert_eq!(
            reason("http://api.openai.com./", &stub),
            NotLocalReason::CloudHost
        );
        assert_eq!(stub.calls.load(Ordering::SeqCst), 0);
        stub.map.insert("evil.com", Ok(vec![ip("1.2.3.4")]));
        assert_eq!(
            reason("evil.com:443", &stub),
            NotLocalReason::PublicAddress(ip("1.2.3.4"))
        );
        assert_eq!(stub.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn mapped_v4_follows_the_v4_rules() {
        let stub = Stub::default();
        assert_eq!(
            reason("http://[::ffff:8.8.8.8]/", &stub),
            NotLocalReason::PublicAddress(ip("::ffff:8.8.8.8"))
        );
        assert!(require_local_with("http://[::ffff:127.0.0.1]/", &stub).is_ok());
    }

    #[test]
    fn display_text_for_cloud_and_unresolved() {
        let stub = Stub::default();
        let tail = " — aivyx is local-only; point it at a model server on this machine or your own network";
        assert_eq!(
            require_local_with("https://api.openai.com/v1", &stub)
                .unwrap_err()
                .to_string(),
            format!("https://api.openai.com/v1 is not local (cloud host){tail}")
        );
        assert_eq!(
            require_local_with("http://nope.example.com/", &stub)
                .unwrap_err()
                .to_string(),
            format!("http://nope.example.com/ is not local (doesn't resolve){tail}")
        );
    }

    #[test]
    fn resolve_local_with_checks_every_address() {
        let mut stub = Stub::default();
        stub.map.insert(
            "lan.example.com",
            Ok(vec![
                "10.0.0.5".parse().unwrap(),
                "192.168.1.2".parse().unwrap(),
            ]),
        );
        stub.map.insert(
            "mixed.example.com",
            Ok(vec![
                "10.0.0.5".parse().unwrap(),
                "8.8.8.8".parse().unwrap(),
            ]),
        );
        stub.map.insert("empty.example.com", Ok(vec![]));
        let ok = resolve_local_with("lan.example.com", 8080, &stub).unwrap();
        assert_eq!(ok.len(), 2);
        assert!(ok.iter().all(|a| a.port() == 8080));
        let e = resolve_local_with("mixed.example.com", 80, &stub).unwrap_err();
        assert_eq!(
            e.reason,
            NotLocalReason::PublicAddress("8.8.8.8".parse().unwrap())
        );
        assert_eq!(e.url, "mixed.example.com:80");
        for h in ["empty.example.com", "unknown.example.com"] {
            let e = resolve_local_with(h, 80, &stub).unwrap_err();
            assert_eq!(e.reason, NotLocalReason::DoesNotResolve, "{h}");
        }
    }

    #[test]
    fn resolve_local_with_cloud_hosts_and_literals_need_no_lookup() {
        // A stub that knows nothing: any lookup would be DoesNotResolve.
        let stub = Stub::default();
        let e = resolve_local_with("api.openai.com", 443, &stub).unwrap_err();
        assert_eq!(e.reason, NotLocalReason::CloudHost);
        let ok = resolve_local_with("127.0.0.1", 1, &stub).unwrap();
        assert_eq!(ok, vec!["127.0.0.1:1".parse().unwrap()]);
        let ok = resolve_local_with("[::1]", 2, &stub).unwrap();
        assert_eq!(ok, vec!["[::1]:2".parse().unwrap()]);
        assert_eq!(
            resolve_local_with("8.8.8.8", 1, &stub).unwrap_err().reason,
            NotLocalReason::PublicAddress("8.8.8.8".parse().unwrap())
        );
        assert_eq!(stub.calls.load(Ordering::SeqCst), 0, "no lookups");
    }
}
