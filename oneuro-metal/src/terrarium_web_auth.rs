//! Simple bearer-token authentication for the terrarium web app.
//!
//! When `--require-auth` is passed, mutation endpoints require a valid token.
//! Tokens are generated via GET /api/auth/token and expire after 24 hours.

use serde::Serialize;
use std::collections::HashMap;

/// Token metadata.
#[derive(Debug, Clone)]
pub struct TokenInfo {
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
}

/// Authentication state.
pub struct AuthState {
    tokens: HashMap<String, TokenInfo>,
    pub require_auth: bool,
    counter: u64,
}

impl AuthState {
    pub fn new(require_auth: bool) -> Self {
        Self {
            tokens: HashMap::new(),
            require_auth,
            counter: 0,
        }
    }

    /// Generate a new 32-char hex token with 24h expiry.
    pub fn generate_token(&mut self) -> TokenResponse {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.counter += 1;
        let hasher_input = format!("{}-{}-{}", now_ms, self.counter, std::process::id());
        let token = simple_hex_hash(&hasher_input);

        let expires_at_ms = now_ms + 86_400_000; // 24 hours
        let info = TokenInfo {
            created_at_ms: now_ms,
            expires_at_ms,
        };
        self.tokens.insert(token.clone(), info);

        // Prune expired tokens
        self.tokens.retain(|_, v| v.expires_at_ms > now_ms);

        TokenResponse {
            token,
            expires_in: 86400,
        }
    }

    /// Validate a bearer token.
    pub fn validate_token(&self, token: &str) -> bool {
        if !self.require_auth {
            return true;
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.tokens
            .get(token)
            .map(|info| info.expires_at_ms > now_ms)
            .unwrap_or(false)
    }

    /// Check if a request needs auth and extract token from Authorization header.
    pub fn check_auth(&self, auth_header: Option<&str>) -> bool {
        if !self.require_auth {
            return true;
        }
        match auth_header {
            Some(header) => {
                if let Some(token) = header.strip_prefix("Bearer ") {
                    self.validate_token(token)
                } else {
                    false
                }
            }
            None => false,
        }
    }
}

/// Simple deterministic hash for token generation (not cryptographic, but sufficient for local use).
fn simple_hex_hash(input: &str) -> String {
    let mut h: u64 = 5381;
    for byte in input.bytes() {
        h = h.wrapping_mul(33).wrapping_add(byte as u64);
    }
    let h2 = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    format!("{:016x}{:016x}", h, h2)
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenResponse {
    pub token: String,
    pub expires_in: u64,
}
