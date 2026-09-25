use super::*;
use std::{
    net::{TcpStream, ToSocketAddrs},
    sync::Mutex,
    time::Duration,
};
use tungstenite::{
    client::IntoClientRequest, protocol::WebSocketConfig, stream::MaybeTlsStream, Message,
    WebSocket,
};

pub(super) struct Connection {
    socket: Option<WebSocket<MaybeTlsStream<TcpStream>>>,
    profiling: Option<bool>,
}
impl HttpFfnShards {
    /// Experimental persistent exact-f32 WebSocket transport. HTTP(S) roots
    /// select WS(S); TLS verification and bearer authorization remain enabled.
    pub fn connect_stream(urls: &[String], token: Option<&str>) -> Result<Self, String> {
        let mut client = Self::connect_binary(urls, token)?;
        let mut streams = Vec::with_capacity(urls.len());
        for (root, binding) in urls.iter().zip(&client.bindings) {
            let mut url = reqwest::Url::parse(root).map_err(|e| e.to_string())?;
            let scheme = match url.scheme() {
                "http" => "ws",
                "https" => "wss",
                _ => return Err("FFN stream requires HTTP(S) root".into()),
            };
            url.set_scheme(scheme).map_err(|_| "invalid stream URL")?;
            url.set_path(&format!(
                "{}{}",
                url.path().trim_end_matches('/'),
                binary::STREAM_PATH
            ));
            let host = url.host_str().ok_or("stream host missing")?;
            let port = url.port_or_known_default().ok_or("stream port missing")?;
            let timeout = Duration::from_secs(60);
            let mut tcp = None;
            for address in (host, port).to_socket_addrs().map_err(|e| e.to_string())? {
                if let Ok(socket) = TcpStream::connect_timeout(&address, timeout) {
                    tcp = Some(socket);
                    break;
                }
            }
            let tcp = tcp.ok_or("FFN stream connection failed")?;
            tcp.set_read_timeout(Some(timeout))
                .map_err(|e| e.to_string())?;
            tcp.set_write_timeout(Some(timeout))
                .map_err(|e| e.to_string())?;
            tcp.set_nodelay(true).map_err(|e| e.to_string())?;
            let mut request = url
                .as_str()
                .into_client_request()
                .map_err(|e| e.to_string())?;
            request.headers_mut().insert(
                "Sec-WebSocket-Protocol",
                binary::STREAM_PROTOCOL.parse().unwrap(),
            );
            if let Some(token) = token {
                let mut value =
                    tungstenite::http::HeaderValue::from_str(&format!("Bearer {token}"))
                        .map_err(|e| e.to_string())?;
                value.set_sensitive(true);
                request.headers_mut().insert("Authorization", value);
            }
            let limit = binding
                .program
                .hidden
                .checked_mul(4)
                .and_then(|n| n.checked_add(binary::HEADER_BYTES))
                .ok_or("FFN frame size overflow")?
                .max(binary::CONTROL_LIMIT);
            let config = WebSocketConfig::default()
                .max_message_size(Some(limit))
                .max_frame_size(Some(limit));
            let (socket, response) =
                tungstenite::client_tls_with_config(request, tcp, Some(config), None)
                    .map_err(|e| e.to_string())?;
            if response
                .headers()
                .get("Sec-WebSocket-Protocol")
                .is_none_or(|v| v != binary::STREAM_PROTOCOL)
            {
                return Err("FFN stream subprotocol mismatch".into());
            }
            streams.push(Mutex::new(Connection {
                socket: Some(socket),
                profiling: None,
            }));
        }
        client.streams = Some(streams);
        Ok(client)
    }
    pub(super) fn forward_stream(
        &self,
        shard: usize,
        layer: usize,
        row: &[f32],
    ) -> Result<Vec<f32>, String> {
        let started = profile::enabled().then(Instant::now);
        let mut record = started.map(|_| serde_json::json!({"wire": "stream-f32-v1", "shard": shard, "layer": layer, "complete": false}));
        let mut connection = self
            .streams
            .as_ref()
            .and_then(|s| s.get(shard))
            .ok_or("unknown FFN stream")?
            .lock()
            .map_err(|_| "FFN stream lock poisoned")?;
        // Take ownership: only a fully validated response restores the socket.
        // Timeout, decode failure or correlation error makes reuse impossible.
        let mut socket = connection
            .socket
            .take()
            .ok_or("FFN stream failed; create a fresh coordinator")?;
        let result = (|| {
            let profiling = started.is_some();
            if connection.profiling.is_some_and(|p| p != profiling) {
                return Err("FFN stream profiling mode cannot change".into());
            }
            if connection.profiling.is_none() {
                let options = serde_json::to_string(&binary::StreamOptions { profile: profiling })
                    .map_err(|e| e.to_string())?;
                if let Some(record) = &mut record {
                    record["stream_setup_bytes"] =
                        (options.len() + binary::websocket_overhead(options.len(), true)).into();
                }
                socket
                    .send(Message::Text(options.into()))
                    .map_err(|e| e.to_string())?;
                connection.profiling = Some(profiling);
            }
            let handle = self.handles.as_ref().unwrap()[shard];
            let sequence = self.sequences[shard]
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .map_err(|_| "FFN sequence exhausted")?;
            let encode = started.map(|_| Instant::now());
            let body = binary::encode(Direction::Request, handle, sequence, layer, row)?;
            let payload_len = body.len();
            if let (Some(record), Some(encode)) = (&mut record, encode) {
                record["encode_ns"] = (encode.elapsed().as_nanos() as u64).into();
                record["request_bytes"] = payload_len.into();
            }
            let roundtrip = started.map(|_| Instant::now());
            socket
                .send(Message::Binary(body.into()))
                .map_err(|e| e.to_string())?;
            let mut worker = None;
            let mut overhead = binary::websocket_overhead(payload_len, true);
            if profiling {
                let Message::Text(text) = socket.read().map_err(|e| e.to_string())? else {
                    return Err("FFN stream timing missing".into());
                };
                if text.len() > binary::CONTROL_LIMIT {
                    return Err("oversize FFN timing".into());
                }
                let metadata: binary::StreamTiming =
                    serde_json::from_str(&text).map_err(|e| e.to_string())?;
                if metadata.sequence != sequence {
                    return Err("FFN stream timing sequence mismatch".into());
                }
                overhead += binary::websocket_overhead(text.len(), false);
                record.as_mut().unwrap()["telemetry_bytes"] = text.len().into();
                worker = Some(metadata.timing);
            }
            let Message::Binary(body) = socket.read().map_err(|e| e.to_string())? else {
                return Err("FFN stream response must be binary".into());
            };
            overhead += binary::websocket_overhead(body.len(), false);
            if let (Some(record), Some(roundtrip)) = (&mut record, roundtrip) {
                let ns = roundtrip.elapsed().as_nanos() as u64;
                record["roundtrip_ns"] = ns.into();
                record["response_bytes"] = body.len().into();
                record["websocket_overhead_bytes"] = overhead.into();
                if let Some(worker) = worker {
                    record["transport_remainder_ns"] = ns.checked_sub(worker.handler_ns).into();
                    record["worker"] = serde_json::to_value(worker).map_err(|e| e.to_string())?;
                }
            }
            let decode = started.map(|_| Instant::now());
            let frame = binary::decode(
                &body,
                Direction::Response,
                self.bindings[shard].program.hidden,
            )?;
            if frame.handle != handle || frame.sequence != sequence || frame.layer as usize != layer
            {
                return Err("FFN stream response correlation mismatch".into());
            }
            if let (Some(record), Some(decode)) = (&mut record, decode) {
                record["decode_ns"] = (decode.elapsed().as_nanos() as u64).into();
            }
            Ok(frame.row)
        })();
        if result.is_ok() {
            connection.socket = Some(socket);
        }
        if let (Some(mut record), Some(started)) = (record, started) {
            record["complete"] = result.is_ok().into();
            record["total_ns"] = (started.elapsed().as_nanos() as u64).into();
            profile::record_provider_call(record);
        }
        result
    }
}
