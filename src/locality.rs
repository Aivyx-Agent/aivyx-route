//! Where an endpoint's address points: on this machine or network
//! (`Local`), or anywhere else (`Cloud`). Pure string inspection — no DNS.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use crate::profile::Locality;

/// The locality an endpoint URL implies, from its host alone.
///
/// `Local` for an IP literal in a loopback, private (RFC 1918, ULA
/// `fc00::/7`), link-local, CGNAT/Tailscale (`100.64.0.0/10`) or
/// unspecified range, and for a hostname that is `localhost` (or ends in
/// `.localhost`), has no dots, or ends in `.local`, `.lan`, `.internal` or
/// `.home.arpa`. `Cloud` for every other host, and for a URL with no host
/// (fail closed).
pub fn url_locality(url: &str) -> Locality {
    url_host(url).map_or(Locality::Cloud, |host| host_locality(&host))
}

/// The lowercased host of `url`, without brackets, port, userinfo or a
/// trailing dot. `None` when there is no host.
pub(crate) fn url_host(url: &str) -> Option<String> {
    let url = url.trim();
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = rest.split(['/', '\\', '?', '#']).next().unwrap_or("");
    let host_port = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    let host = match host_port.strip_prefix('[') {
        Some(bracketed) => bracketed.split_once(']')?.0,
        None => host_port.split(':').next().unwrap_or(""),
    };
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    (!host.is_empty()).then_some(host)
}

fn host_locality(host: &str) -> Locality {
    if let Some(ip) = parse_ip(host) {
        return if ip_is_local(ip) {
            Locality::Local
        } else {
            Locality::Cloud
        };
    }
    let local = host == "localhost"
        || !host.contains('.')
        || [".localhost", ".local", ".lan", ".internal", ".home.arpa"]
            .iter()
            .any(|suffix| host.ends_with(suffix));
    if local {
        Locality::Local
    } else {
        Locality::Cloud
    }
}

/// An IP literal, including the single-number IPv4 forms URL parsers
/// accept (`2130706433`, `0x7f000001`), so a dotless number isn't mistaken
/// for a LAN hostname.
fn parse_ip(host: &str) -> Option<IpAddr> {
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Some(ip);
    }
    let n = match host.strip_prefix("0x") {
        Some(hex) => u32::from_str_radix(hex, 16).ok()?,
        None if host.bytes().all(|b| b.is_ascii_digit()) => match host.parse::<u32>() {
            Ok(n) => n,
            // A number too large for IPv4 is no address at all.
            Err(_) => return Some(IpAddr::V4(Ipv4Addr::BROADCAST)),
        },
        None => return None,
    };
    Some(IpAddr::V4(Ipv4Addr::from(n)))
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
        assert_eq!(url_host("http:///p"), None);
    }
}
