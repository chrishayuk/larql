//! Blocking transport for the V3 dense FFN operation provider.
use larql_inference::vindex3::dense_ffn::{profile, FfnTransport};
use larql_router_protocol::vindex3_ffn::binary::{self, Direction};
use larql_router_protocol::vindex3_ffn::{
    Binding, Request, Response, WorkerTiming, PATH, PROFILE_HEADER,
};
use std::{
    io::Read,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

pub struct HttpFfnShards {
    client: reqwest::blocking::Client,
    urls: Vec<String>,
    bindings: Vec<Binding>,
    handles: Option<Vec<binary::Handle>>,
    sequences: Vec<AtomicU64>,
    streams: Option<Vec<std::sync::Mutex<stream::Connection>>>,
}
impl HttpFfnShards {
    pub fn connect(urls: &[String], token: Option<&str>) -> Result<Self, String> {
        Self::connect_with(urls, token, false)
    }
    pub fn connect_binary(urls: &[String], token: Option<&str>) -> Result<Self, String> {
        Self::connect_with(urls, token, true)
    }
    fn connect_with(
        urls: &[String],
        token: Option<&str>,
        binary_wire: bool,
    ) -> Result<Self, String> {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(token) = token {
            let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
                .map_err(|e| e.to_string())?;
            value.set_sensitive(true);
            headers.insert(reqwest::header::AUTHORIZATION, value);
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .redirect(reqwest::redirect::Policy::none())
            .pool_idle_timeout(None)
            .pool_max_idle_per_host(1)
            .default_headers(headers)
            .build()
            .map_err(|e| e.to_string())?;
        let mut urls: Vec<_> = urls
            .iter()
            .map(|u| format!("{}{PATH}", u.trim_end_matches('/')))
            .collect();
        let bindings = urls
            .iter()
            .map(|u| {
                client
                    .get(u)
                    .send()
                    .and_then(|r| r.error_for_status())
                    .and_then(|r| r.json::<Binding>())
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let handles = if binary_wire {
            let mut handles = Vec::with_capacity(urls.len());
            for (url, binding) in urls.iter_mut().zip(&bindings) {
                binding.validate()?;
                let root = url.strip_suffix(PATH).expect("FFN URL");
                let opened = client
                    .post(format!("{root}{}", binary::OPEN_PATH))
                    .json(binding)
                    .send()
                    .and_then(|r| r.error_for_status())
                    .and_then(|r| r.json::<binary::Opened>())
                    .map_err(|e| format!("binary FFN open failed: {e}"))?;
                if opened.version != binary::VERSION || opened.binding != *binding {
                    return Err("binary FFN open changed binding or wire version".into());
                }
                handles.push(opened.handle);
                *url = format!("{root}{}", binary::PATH);
            }
            Some(handles)
        } else {
            None
        };
        let sequences = urls.iter().map(|_| AtomicU64::new(1)).collect();
        Ok(Self {
            client,
            urls,
            bindings,
            handles,
            sequences,
            streams: None,
        })
    }
}
impl FfnTransport for HttpFfnShards {
    fn bindings(&self) -> Vec<Binding> {
        self.bindings.clone()
    }
    fn forward_bound(
        &self,
        shard: usize,
        layer: usize,
        normalized: &[f32],
        expected: &Binding,
    ) -> Result<Vec<f32>, String> {
        if self.handles.is_some() {
            return self.forward_binary(shard, layer, normalized);
        }
        let response = self.forward(shard, layer, normalized)?;
        if response.binding != *expected || response.layer != layer {
            return Err("dense FFN response changed binding or layer".into());
        }
        Ok(response.row)
    }
    fn forward(&self, shard: usize, layer: usize, normalized: &[f32]) -> Result<Response, String> {
        let url = self.urls.get(shard).ok_or("unknown V3 shard index")?;
        let binding = self.bindings.get(shard).ok_or("unknown V3 shard binding")?;
        if self.handles.is_some() {
            return Ok(Response {
                binding: binding.clone(),
                layer,
                row: self.forward_binary(shard, layer, normalized)?,
            });
        }
        if profile::enabled() {
            return self.forward_profiled(url, binding, shard, layer, normalized);
        }
        self.client
            .post(url)
            .json(&Request {
                binding: binding.clone(),
                layer,
                row: normalized.to_vec(),
            })
            .send()
            .and_then(|r| r.error_for_status())
            .and_then(|r| r.json())
            .map_err(|e| e.to_string())
    }
}

impl HttpFfnShards {
    fn forward_profiled(
        &self,
        url: &str,
        binding: &Binding,
        shard: usize,
        layer: usize,
        normalized: &[f32],
    ) -> Result<Response, String> {
        let started = Instant::now();
        let mut record = serde_json::json!({"shard": shard, "layer": layer, "complete": false});
        let result = (|| {
            let encode = Instant::now();
            let body = serde_json::to_vec(&Request {
                binding: binding.clone(),
                layer,
                row: normalized.to_vec(),
            })
            .map_err(|e| e.to_string())?;
            record["encode_ns"] = (encode.elapsed().as_nanos() as u64).into();
            record["request_bytes"] = body.len().into();
            let roundtrip = Instant::now();
            let response = self
                .client
                .post(url)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .header(PROFILE_HEADER, "1")
                .body(body)
                .send()
                .map_err(|e| e.to_string())?;
            let status = response.status();
            // An old worker can omit this header. Never invent zero server time.
            let worker = response
                .headers()
                .get(PROFILE_HEADER)
                .and_then(|h| serde_json::from_slice::<WorkerTiming>(h.as_bytes()).ok());
            let body = response.bytes().map_err(|e| e.to_string())?;
            let roundtrip_ns = roundtrip.elapsed().as_nanos() as u64;
            record["roundtrip_ns"] = roundtrip_ns.into();
            record["response_bytes"] = body.len().into();
            record["http_status"] = status.as_u16().into();
            if let Some(worker) = worker {
                record["transport_remainder_ns"] =
                    roundtrip_ns.checked_sub(worker.handler_ns).into();
                record["worker"] = serde_json::to_value(worker).map_err(|e| e.to_string())?;
            }
            if !status.is_success() {
                return Err(format!("dense FFN HTTP {status}"));
            }
            let decode = Instant::now();
            let response = serde_json::from_slice(&body).map_err(|e| e.to_string());
            record["decode_ns"] = (decode.elapsed().as_nanos() as u64).into();
            response
        })();
        record["total_ns"] = (started.elapsed().as_nanos() as u64).into();
        record["complete"] = result.is_ok().into();
        profile::record_provider_call(record);
        result
    }
}

impl HttpFfnShards {
    fn forward_binary(
        &self,
        shard: usize,
        layer: usize,
        normalized: &[f32],
    ) -> Result<Vec<f32>, String> {
        let handle = *self
            .handles
            .as_ref()
            .and_then(|h| h.get(shard))
            .ok_or("unknown binary FFN handle")?;
        let hidden = self.bindings[shard].program.hidden;
        if normalized.len() != hidden {
            return Err("binary FFN input width mismatch".into());
        }
        if self.streams.is_some() {
            return self.forward_stream(shard, layer, normalized);
        }
        let sequence = self.sequences[shard]
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| "FFN sequence exhausted; reconnect")?;
        let mut record = profile::enabled().then(|| serde_json::json!({"wire": "binary-f32-v1", "shard": shard, "layer": layer, "complete": false}));
        let started = record.as_ref().map(|_| Instant::now());
        let result = (|| {
            let body = binary::encode(Direction::Request, handle, sequence, layer, normalized)?;
            if let (Some(record), Some(started)) = (&mut record, started) {
                record["encode_ns"] = (started.elapsed().as_nanos() as u64).into();
                record["request_bytes"] = body.len().into();
            }
            let roundtrip = started.map(|_| Instant::now());
            let mut request = self
                .client
                .post(&self.urls[shard])
                .header(reqwest::header::CONTENT_TYPE, binary::CONTENT_TYPE)
                .body(body);
            if started.is_some() {
                request = request.header(PROFILE_HEADER, "1");
            }
            let response = request
                .send()
                .and_then(|r| r.error_for_status())
                .map_err(|e| e.to_string())?;
            if response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .is_none_or(|v| v != binary::CONTENT_TYPE)
            {
                return Err("binary FFN response content type mismatch".into());
            }
            let worker = response
                .headers()
                .get(PROFILE_HEADER)
                .and_then(|h| serde_json::from_slice::<WorkerTiming>(h.as_bytes()).ok());
            let expected_len = hidden
                .checked_mul(4)
                .and_then(|n| n.checked_add(binary::HEADER_BYTES))
                .ok_or("FFN frame size overflow")?;
            let mut body = Vec::with_capacity(expected_len);
            response
                .take(expected_len as u64 + 1)
                .read_to_end(&mut body)
                .map_err(|e| e.to_string())?;
            if let (Some(record), Some(roundtrip)) = (&mut record, roundtrip) {
                let ns = roundtrip.elapsed().as_nanos() as u64;
                record["roundtrip_ns"] = ns.into();
                record["response_bytes"] = body.len().into();
                record["http_status"] = 200.into();
                if let Some(worker) = worker {
                    record["transport_remainder_ns"] = ns.checked_sub(worker.handler_ns).into();
                    record["worker"] = serde_json::to_value(worker).map_err(|e| e.to_string())?;
                }
            }
            let decode = started.map(|_| Instant::now());
            let frame = binary::decode(&body, Direction::Response, hidden)?;
            if frame.handle != handle || frame.sequence != sequence || frame.layer as usize != layer
            {
                return Err("binary FFN response handle, sequence or layer mismatch".into());
            }
            if let (Some(record), Some(decode)) = (&mut record, decode) {
                record["decode_ns"] = (decode.elapsed().as_nanos() as u64).into();
            }
            Ok(frame.row)
        })();
        if let (Some(mut record), Some(started)) = (record, started) {
            record["complete"] = result.is_ok().into();
            record["total_ns"] = (started.elapsed().as_nanos() as u64).into();
            profile::record_provider_call(record);
        }
        result
    }
}

#[cfg(test)]
mod tests;

mod stream;
