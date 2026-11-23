//! With `HTTP_PROXY`/`HTTPS_PROXY` pointing at a closed port, a request from
//! `discovery::client()` to a local server still succeeds: `client()` sets
//! `no_proxy`.
//!
//! Its own test binary: it sets process-wide proxy variables.
#![cfg(feature = "discovery")]

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[tokio::test]
async fn the_discovery_client_reaches_a_local_server_despite_a_dead_proxy() {
    let closed = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap()
    };
    // SAFETY: this binary runs this one test; nothing else touches the
    // environment concurrently.
    unsafe {
        for var in ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"] {
            std::env::set_var(var, format!("http://{closed}"));
        }
        std::env::remove_var("NO_PROXY");
        std::env::remove_var("no_proxy");
    }
    let server = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/ok", server.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut socket, _) = server.accept().await.unwrap();
        let mut buf = [0u8; 1024];
        let _ = socket.read(&mut buf).await;
        let _ = socket
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nhi")
            .await;
    });
    let client = aivyx_route::discovery::client().unwrap();
    let body = client.get(&url).send().await.unwrap().text().await.unwrap();
    assert_eq!(body, "hi");
}
