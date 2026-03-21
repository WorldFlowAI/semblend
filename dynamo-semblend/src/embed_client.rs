//! HTTP client for the Python MiniLM embedding sidecar.
//!
//! Communicates with the sidecar via localhost HTTP.
//! Protocol: POST /embed {"text": "..."} -> {"embedding": [f32; 384]}

use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing;

const EMBEDDING_DIM: usize = 384;

#[derive(Serialize)]
struct EmbedRequest {
    text: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

/// Client for the Python MiniLM embedding sidecar.
#[derive(Clone)]
pub struct EmbedClient {
    url: String,
    client: reqwest::Client,
}

impl EmbedClient {
    pub fn new(url: &str) -> Self {
        Self {
            url: url.trim_end_matches('/').to_string(),
            client: reqwest::Client::builder()
                .pool_max_idle_per_host(4)
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("Failed to build HTTP client"),
        }
    }

    /// Embed text and return a 384-dim normalized f32 vector.
    /// On failure, returns None (graceful degradation).
    pub async fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let t0 = Instant::now();

        let sorted_text = order_invariant_text(text);

        let resp = self
            .client
            .post(format!("{}/embed", self.url))
            .json(&EmbedRequest {
                text: sorted_text,
            })
            .send()
            .await
            .ok()?;

        let body: EmbedResponse = resp.json().await.ok()?;

        if body.embedding.len() != EMBEDDING_DIM {
            tracing::warn!(
                "Expected {EMBEDDING_DIM}-dim embedding, got {}",
                body.embedding.len()
            );
            return None;
        }

        let elapsed = t0.elapsed();
        if elapsed.as_millis() > 10 {
            tracing::debug!(ms = elapsed.as_millis(), "Embed sidecar latency");
        }

        Some(body.embedding)
    }
}

/// Order-invariant text preprocessing.
/// Must match semblend_core.pipeline._order_invariant_text().
fn order_invariant_text(text: &str) -> String {
    let max_chars = 8000;
    let replaced = text.replace('\n', ". ");
    let mut parts: Vec<&str> = replaced
        .split(". ")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    parts.sort_unstable();
    let result = parts.join(". ");
    if result.len() > max_chars {
        result[..max_chars].to_string()
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_invariant() {
        let a = order_invariant_text("Hello world. Foo bar. Baz");
        let b = order_invariant_text("Foo bar. Baz. Hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn test_newlines_replaced() {
        let text = "Line one\nLine two\nLine three";
        let result = order_invariant_text(text);
        assert!(result.contains("Line one"));
        assert!(result.contains("Line two"));
    }
}
