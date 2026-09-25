//! Bind once, exact selected-expert batches over persistent HTTP connections.
use larql_inference::vindex3::dense_ffn::profile;
use larql_inference::vindex3::routed_experts::ExpertOutput;
use larql_inference::vindex3::routed_experts::ExpertTransport;
use larql_router_protocol::vindex3_experts as wire;
use std::{
    io::Read,
    sync::atomic::{AtomicU64, Ordering},
};
pub struct HttpExpertShards {
    client: reqwest::blocking::Client,
    urls: Vec<String>,
    bindings: Vec<wire::Binding>,
    handles: Vec<wire::Handle>,
    sequences: Vec<AtomicU64>,
}
impl HttpExpertShards {
    pub fn connect(urls: &[String], token: Option<&str>) -> Result<Self, String> {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(token) = token {
            let mut h = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
                .map_err(|e| e.to_string())?;
            h.set_sensitive(true);
            headers.insert(reqwest::header::AUTHORIZATION, h);
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .redirect(reqwest::redirect::Policy::none())
            .pool_idle_timeout(None)
            .pool_max_idle_per_host(1)
            .default_headers(headers)
            .build()
            .map_err(|e| e.to_string())?;
        let mut bindings = Vec::new();
        let mut handles = Vec::new();
        let mut paths = Vec::new();
        for url in urls {
            let root = url.trim_end_matches('/');
            let binding: wire::Binding = client
                .get(format!("{root}{}", wire::PATH))
                .send()
                .and_then(|r| r.error_for_status())
                .and_then(|r| r.json())
                .map_err(|e| e.to_string())?;
            binding.validate()?;
            let opened: wire::Opened = client
                .post(format!("{root}{}", wire::OPEN_PATH))
                .json(&binding)
                .send()
                .and_then(|r| r.error_for_status())
                .and_then(|r| r.json())
                .map_err(|e| e.to_string())?;
            if opened.version != 1 || opened.binding != binding {
                return Err("expert open changed binding or protocol version".into());
            }
            bindings.push(binding);
            handles.push(opened.handle);
            paths.push(format!("{root}{}", wire::BINARY_PATH));
        }
        Ok(Self {
            client,
            urls: paths,
            bindings,
            handles,
            sequences: urls.iter().map(|_| AtomicU64::new(1)).collect(),
        })
    }
}
impl ExpertTransport for HttpExpertShards {
    fn bindings(&self) -> Vec<wire::Binding> {
        self.bindings.clone()
    }
    fn forward(
        &self,
        shard: usize,
        layer: usize,
        experts: &[usize],
        row: &[f32],
    ) -> Result<Vec<ExpertOutput>, String> {
        let started = profile::enabled().then(std::time::Instant::now);
        let mut record = started.map(|_| {
            serde_json::json!({
                "wire": "experts-binary-f32-v1", "shard": shard, "layer": layer,
                "selected_count": experts.len(), "complete": false,
            })
        });
        let result = (|| {
            let binding = self.bindings.get(shard).ok_or("unknown expert worker")?;
            if row.len() != binding.program.hidden {
                return Err("expert input width mismatch".into());
            }
            let handle = self.handles[shard];
            let sequence = self.sequences[shard]
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |s| s.checked_add(1))
                .map_err(|_| "expert sequence exhausted")?;
            let encode = started.map(|_| std::time::Instant::now());
            let body = wire::encode_request(handle, sequence, layer, experts, row)?;
            if let (Some(record), Some(encode)) = (&mut record, encode) {
                record["encode_ns"] = (encode.elapsed().as_nanos() as u64).into();
                record["request_bytes"] = body.len().into();
            }
            let roundtrip = started.map(|_| std::time::Instant::now());
            let mut request = self
                .client
                .post(&self.urls[shard])
                .header(reqwest::header::CONTENT_TYPE, wire::CONTENT_TYPE)
                .body(body);
            if started.is_some() {
                request = request.header(wire::PROFILE_HEADER, "1");
            }
            let response = request
                .send()
                .and_then(|r| r.error_for_status())
                .map_err(|e| e.to_string())?;
            if response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .is_none_or(|h| h != wire::CONTENT_TYPE)
            {
                return Err("expert response content type mismatch".into());
            }
            let worker = started
                .and_then(|_| response.headers().get(wire::PROFILE_HEADER))
                .and_then(|h| serde_json::from_slice::<wire::WorkerTiming>(h.as_bytes()).ok());
            let limit = wire::response_len(row.len(), experts.len())?;
            let mut bytes = Vec::with_capacity(limit);
            response
                .take(limit as u64 + 1)
                .read_to_end(&mut bytes)
                .map_err(|e| e.to_string())?;
            if let (Some(record), Some(roundtrip)) = (&mut record, roundtrip) {
                let ns = roundtrip.elapsed().as_nanos() as u64;
                record["roundtrip_ns"] = ns.into();
                record["response_bytes"] = bytes.len().into();
                record["worker_profile_complete"] = worker.is_some().into();
                if let Some(worker) = worker {
                    record["transport_remainder_ns"] = ns.checked_sub(worker.handler_ns).into();
                    record["worker"] = serde_json::to_value(worker).map_err(|e| e.to_string())?;
                }
            }
            let decode = started.map(|_| std::time::Instant::now());
            let response = wire::decode_response(&bytes, row.len(), experts.len())?;
            if response.handle != handle || response.sequence != sequence || response.layer != layer
            {
                return Err("expert response handle, sequence or layer mismatch".into());
            }
            let mut seen = std::collections::BTreeSet::new();
            if response.rows.len() != experts.len()
                || response
                    .rows
                    .iter()
                    .any(|(id, _)| !experts.contains(id) || !seen.insert(*id))
            {
                return Err("expert response IDs or count mismatch".into());
            }
            if let (Some(record), Some(decode)) = (&mut record, decode) {
                record["decode_ns"] = (decode.elapsed().as_nanos() as u64).into();
            }
            Ok(response
                .rows
                .into_iter()
                .map(|(expert, row)| ExpertOutput { expert, row })
                .collect())
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
