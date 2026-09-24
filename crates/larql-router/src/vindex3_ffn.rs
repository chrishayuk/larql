//! Blocking transport for the V3 dense FFN operation provider.
use larql_inference::vindex3::dense_ffn::{profile, FfnTransport};
use larql_router_protocol::vindex3_ffn::{
    Binding, Request, Response, WorkerTiming, PATH, PROFILE_HEADER,
};
use std::time::Instant;

pub struct HttpFfnShards {
    client: reqwest::blocking::Client,
    urls: Vec<String>,
    bindings: Vec<Binding>,
}
impl HttpFfnShards {
    pub fn connect(urls: &[String], token: Option<&str>) -> Result<Self, String> {
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
            .default_headers(headers)
            .build()
            .map_err(|e| e.to_string())?;
        let urls: Vec<_> = urls
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
        Ok(Self {
            client,
            urls,
            bindings,
        })
    }
}
impl FfnTransport for HttpFfnShards {
    fn bindings(&self) -> Vec<Binding> {
        self.bindings.clone()
    }
    fn forward(&self, shard: usize, layer: usize, normalized: &[f32]) -> Result<Response, String> {
        let url = self.urls.get(shard).ok_or("unknown V3 shard index")?;
        let binding = self.bindings.get(shard).ok_or("unknown V3 shard binding")?;
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
