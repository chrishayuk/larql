//! A connection pins one immutable worker incarnation. No KV or retry state.
use super::*;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use std::time::Duration;

pub async fn upgrade(
    State(state): State<Arc<AppState>>,
    ws: WebSocketUpgrade,
) -> Result<Response, ServerError> {
    let model = worker(&state)?;
    let wire = model
        .ffn_wire
        .as_ref()
        .ok_or_else(|| ServerError::Unsupported("FFN stream unavailable".into()))?;
    let limit = wire
        .execution
        .binding()
        .program
        .hidden
        .checked_mul(4)
        .and_then(|n| n.checked_add(binary::HEADER_BYTES))
        .ok_or_else(|| ServerError::BadRequest("FFN frame size overflow".into()))?
        .max(binary::CONTROL_LIMIT);
    Ok(ws
        .protocols([binary::STREAM_PROTOCOL])
        .max_frame_size(limit)
        .max_message_size(limit)
        .on_upgrade(move |socket| async move {
            // Any malformed frame, failed operation or disconnected peer ends
            // the stream. Never send a substitute numerical contribution.
            if let Err(error) = serve(socket, model, state).await {
                tracing::debug!(%error, "V3 FFN stream ended");
            }
        }))
}

async fn receive(socket: &mut WebSocket) -> Result<Message, String> {
    tokio::time::timeout(Duration::from_secs(60), socket.recv())
        .await
        .map_err(|_| "FFN stream receive timeout")?
        .ok_or("FFN stream closed")?
        .map_err(|e| e.to_string())
}
async fn send(socket: &mut WebSocket, message: Message) -> Result<(), String> {
    tokio::time::timeout(Duration::from_secs(60), socket.send(message))
        .await
        .map_err(|_| "FFN stream send timeout")?
        .map_err(|e| e.to_string())
}
async fn serve(
    mut socket: WebSocket,
    model: Arc<V3Model>,
    state: Arc<AppState>,
) -> Result<(), String> {
    let Message::Text(options) = receive(&mut socket).await? else {
        return Err("stream requires initial options".into());
    };
    if options.len() > binary::CONTROL_LIMIT {
        return Err("oversize stream options".into());
    }
    let options: binary::StreamOptions =
        serde_json::from_str(&options).map_err(|e| e.to_string())?;
    let wire = model.ffn_wire.as_ref().expect("checked worker");
    let mut last_sequence = 0;
    loop {
        let message = receive(&mut socket).await?;
        let started = options.profile.then(Instant::now);
        let Message::Binary(bytes) = message else {
            return Err("stream requires binary FFN frame".into());
        };
        let frame = binary::decode(
            &bytes,
            Direction::Request,
            wire.execution.binding().program.hidden,
        )?;
        if frame.handle != wire.handle || frame.sequence <= last_sequence {
            return Err("stale FFN handle or non-increasing sequence".into());
        }
        let range = &wire.execution.binding().program;
        if !(range.start..range.end).contains(&(frame.layer as usize)) {
            return Err("layer outside bound FFN worker".into());
        }
        last_sequence = frame.sequence;
        state.bump_requests();
        let mut timing = WorkerTiming::default();
        if let Some(started) = started {
            timing.decode_ns = started.elapsed().as_nanos() as u64;
        }
        let queued = started.map(|_| Instant::now());
        let worker = model.clone();
        let (row, mut timing) = tokio::task::spawn_blocking(move || {
            if let Some(queued) = queued {
                timing.queue_ns = queued.elapsed().as_nanos() as u64;
            }
            let execute = queued.map(|_| Instant::now());
            let row = worker
                .ffn_wire
                .as_ref()
                .expect("bound worker")
                .execution
                .apply(frame.layer as usize, &frame.row)
                .map_err(|e| e.to_string())?;
            if let Some(execute) = execute {
                timing.ffn_ns = execute.elapsed().as_nanos() as u64;
                timing.execute_ns = timing.ffn_ns;
            }
            Ok::<_, String>((row, timing))
        })
        .await
        .map_err(|e| e.to_string())??;
        let encode = started.map(|_| Instant::now());
        let body = binary::encode(
            Direction::Response,
            frame.handle,
            frame.sequence,
            frame.layer as usize,
            &row,
        )?;
        if let (Some(started), Some(encode)) = (started, encode) {
            timing.encode_ns = encode.elapsed().as_nanos() as u64;
            timing.handler_ns = started.elapsed().as_nanos() as u64;
            let text = serde_json::to_string(&binary::StreamTiming {
                sequence: frame.sequence,
                timing,
            })
            .map_err(|e| e.to_string())?;
            // Diagnostics are a separate correlated text message; the binary
            // numerical frame remains identical to the HTTP control.
            send(&mut socket, Message::Text(text.into())).await?;
        }
        send(&mut socket, Message::Binary(body.into())).await?;
    }
}
