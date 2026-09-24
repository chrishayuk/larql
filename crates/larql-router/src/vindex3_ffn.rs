//! Blocking transport for the V3 dense FFN operation provider.
use larql_inference::vindex3::dense_ffn::FfnTransport;
use larql_router_protocol::vindex3_ffn::{Binding, Request, Response, PATH};

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
