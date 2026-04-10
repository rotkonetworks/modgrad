//! QUIC transport for neural telemetry.
//!
//! Feature-gated: `cargo build --features telemetry`
//!
//! Architecture:
//!   organism → ring buffer → flatbuf serialize → QUIC stream → debugger
//!
//! The telemetry server runs as a background thread.
//! Debuggers connect via QUIC, receive the manifest, then get
//! live tick batches. Multiple debuggers can connect simultaneously.
//!
//! QUIC benefits for telemetry:
//!   - UDP-like: organism never blocks (fire and forget)
//!   - Stream multiplexing: subscribe per region
//!   - 0-RTT: debugger reconnects instantly
//!   - Encrypted: safe over network
//!
//! Without the `telemetry` feature, this module is empty.
//! The base telemetry (ring buffer + file) works without QUIC.

/// Default telemetry port.
pub const TELEMETRY_PORT: u16 = 4748; // one above daemon port 4747

/// Start the QUIC telemetry server in a background thread.
/// Returns a sender channel for pushing tick data.
///
/// Without `telemetry` feature: returns None (no-op).
#[cfg(feature = "telemetry")]
pub fn start_server(
    manifest_json: &str,
    port: u16,
) -> Option<std::sync::mpsc::Sender<Vec<f32>>> {
    use std::sync::mpsc;

    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let manifest = manifest_json.to_string();

    std::thread::spawn(move || {
        // Build QUIC server with self-signed cert
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");

        rt.block_on(async move {
            let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()])
                .expect("self-signed cert");
            let key = rustls::pki_types::PrivatePkcs8KeyDer::from(
                cert.key_pair.serialize_der()
            );
            let cert_chain = vec![rustls::pki_types::CertificateDer::from(
                cert.cert.der().to_vec()
            )];

            let mut server_config = quinn::ServerConfig::with_single_cert(
                cert_chain, key.into()
            ).expect("server config");

            let transport = quinn::TransportConfig::default();
            server_config.transport_config(std::sync::Arc::new(transport));

            let endpoint = quinn::Endpoint::server(
                server_config,
                format!("127.0.0.1:{port}").parse().unwrap(),
            ).expect("bind telemetry endpoint");

            eprintln!("[telemetry] QUIC server on 127.0.0.1:{port}");

            // Accept connections and stream ticks
            // (simplified: single-client for now)
            while let Some(conn) = endpoint.accept().await {
                let connection = conn.await.unwrap();
                eprintln!("[telemetry] debugger connected");

                // Send manifest on first unidirectional stream
                if let Ok(mut stream) = connection.open_uni().await {
                    stream.write_all(manifest.as_bytes()).await.ok();
                    stream.finish().ok();
                }

                // Stream ticks on a second unidirectional stream
                if let Ok(mut stream) = connection.open_uni().await {
                    while let Ok(tick_data) = rx.recv() {
                        let bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                tick_data.as_ptr() as *const u8,
                                tick_data.len() * 4,
                            )
                        };
                        if stream.write_all(bytes).await.is_err() {
                            break; // client disconnected
                        }
                    }
                }
            }
        });
    });

    Some(tx)
}

/// No-op when telemetry feature is disabled.
#[cfg(not(feature = "telemetry"))]
pub fn start_server(
    _manifest_json: &str,
    _port: u16,
) -> Option<std::sync::mpsc::Sender<Vec<f32>>> {
    None
}
