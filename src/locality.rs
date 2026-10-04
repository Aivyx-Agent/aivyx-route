//! Where an endpoint's address points: on this machine or network
//! (`Local`), or anywhere else (`Cloud`). No DNS: the host is classified as
//! the HTTP client will see it, after WHATWG URL parsing (percent-decoding,
//! IDNA, numeric IPv4 forms).

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use url::{Host, Url};

use crate::profile::Locality;

/// The locality an endpoint URL implies, from its host alone.
///
/// The URL is parsed as the HTTP client parses it (`http://` is assumed
/// when there is no scheme). `Local` for an IP in a loopback, private
/// (RFC 1918, ULA `fc00::/7`), link-local, CGNAT/Tailscale
/// (`100.64.0.0/10`) or unspecified range, and for a domain that is
/// `localhost` (or ends in `.localhost`), is a single dotless label, or
/// ends in `.local`, `.lan`, `.internal` or `.home.arpa`. A single label
/// with a trailing dot (`ai.`) is a fully qualified name, so `Cloud`
/// (`localhost.` stays `Local`). `Cloud` for every other host, for a
/// scheme other than `http`/`https`, and for a URL that doesn't parse or
/// has no host (fail closed).
pub fn url_locality(url: &str) -> Locality {
    match parse_host(url) {
        Some(Host::Ipv4(ip)) => locality_if(v4_is_local(ip)),
        Some(Host::Ipv6(ip)) => locality_if(ip_is_local(IpAddr::V6(ip))),
        Some(Host::Domain(domain)) => locality_if(domain_is_local(&domain)),
        None => Locality::Cloud,
    }
}

/// The host of `url` as the HTTP client sees it (no brackets, port or
/// userinfo). `None` when it doesn't parse or has no host.
pub(crate) fn url_host(url: &str) -> Option<String> {
    parse_host(url).map(|host| match host {
        Host::Domain(domain) => domain,
        Host::Ipv4(ip) => ip.to_string(),
        Host::Ipv6(ip) => ip.to_string(),
    })
}

fn parse_host(url: &str) -> Option<Host<String>> {
    let url = url.trim();
    let parsed = if url.contains("://") {
        Url::parse(url)
    } else {
        Url::parse(&format!("http://{url}"))
    }
    .ok()?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return None;
    }
    let host = parsed.host()?.to_owned();
    match &host {
        Host::Domain(domain) if domain.is_empty() => None,
        _ => Some(host),
    }
}

fn locality_if(local: bool) -> Locality {
    if local {
        Locality::Local
    } else {
        Locality::Cloud
    }
}

/// `domain` is already lowercase ASCII (IDNA-normalised) from the parser.
fn domain_is_local(domain: &str) -> bool {
    let fqdn = domain.strip_suffix('.');
    let name = fqdn.unwrap_or(domain);
    if name.is_empty() || name.ends_with('.') {
        return false;
    }
    name == "localhost"
        // A dotless name is a LAN name, unless written fully qualified.
        || (fqdn.is_none() && !name.contains('.'))
        || [".localhost", ".local", ".lan", ".internal", ".home.arpa"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
}

fn ip_is_local(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4_is_local(v4),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => v4_is_local(v4),
            None => v6_is_local(v6),
        },
    }
}

fn v4_is_local(ip: Ipv4Addr) -> bool {
    let [a, b, ..] = ip.octets();
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_unspecified()
        // CGNAT / Tailscale: 100.64.0.0/10.
        || (a == 100 && (b & 0xc0) == 64)
}

fn v6_is_local(ip: Ipv6Addr) -> bool {
    let first = ip.segments()[0];
    ip.is_loopback()
        || ip.is_unspecified()
        // Unique local, fc00::/7.
        || (first & 0xfe00) == 0xfc00
        // Link-local, fe80::/10.
        || (first & 0xffc0) == 0xfe80
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_addresses_are_local() {
        for url in [
            "http://127.0.0.1:8080",
            "http://127.9.9.9",
            "http://10.1.2.3:11434",
            "http://172.16.0.1",
            "http://172.31.255.255",
            "http://192.168.1.20:8080/v1",
            "http://169.254.1.1",
            "http://100.100.100.100:11434",
            "http://100.64.0.1",
            "http://[::1]:8080",
            "http://[fd12:3456::1]/v1",
            "http://[fe80::1]",
            "http://[::ffff:192.168.1.2]",
            "http://0.0.0.0:8080",
            "http://localhost:11434",
            "http://LOCALHOST.:11434",
            "http://gpu.lan:8080",
            "http://box.local",
            "http://llm.internal/v1",
            "http://nas.home.arpa",
            "http://myhost:8080",
            "http://user:pw@myhost:8080",
            "http://2130706433",
            "myhost:8080",
        ] {
            assert_eq!(url_locality(url), Locality::Local, "{url}");
        }
    }

    #[test]
    fn everything_else_is_cloud() {
        for url in [
            "https://api.groq.com/openai/v1",
            "https://openrouter.ai/api/v1",
            "http://8.8.8.8",
            "http://172.32.0.1",
            "http://100.128.0.1",
            "http://192.169.0.1",
            "http://[2001:db8::1]",
            "http://[::ffff:8.8.8.8]",
            "http://134744072",
            "http://99999999999",
            "http://localhost.example.com",
            "http://127.0.0.1.nip.io",
            "http://evil.com#@localhost",
            "http://evil.com\\@localhost",
            "http://localhost@evil.com",
            "",
            "http://",
            // What the HTTP client connects to, not the raw string.
            "http://api%2Egroq%2Ecom/v1",
            "http://8%2E8%2E8%2E8/",
            "http://api\u{3002}groq\u{3002}com/v1",
            "http://api\u{ff0e}groq\u{ff0e}com",
            "http://localhost%2Eevil%2Ecom/",
            "http://0167772161/",
            "http://ai./",
            // Unparseable: fail closed.
            "http://[2001:db8::1%25eth0]/",
            "http://evil.com%2F.lan/",
            "ftp://localhost",
        ] {
            assert_eq!(url_locality(url), Locality::Cloud, "{url}");
        }
    }

    #[test]
    fn hosts_are_extracted_plainly() {
        assert_eq!(
            url_host("https://API.Groq.com:443/x").as_deref(),
            Some("api.groq.com")
        );
        assert_eq!(url_host("http://[::1]:80").as_deref(), Some("::1"));
        assert_eq!(url_host("http://u@h/p").as_deref(), Some("h"));
        // What the client would connect to: WHATWG reads `http:///p` as host `p`.
        assert_eq!(url_host("http:///p").as_deref(), Some("p"));
        assert_eq!(url_host("http://"), None);
        assert_eq!(
            url_host("http://api%2Egroq%2Ecom/v1").as_deref(),
            Some("api.groq.com")
        );
        assert_eq!(url_host("ftp://localhost"), None);
    }
}
