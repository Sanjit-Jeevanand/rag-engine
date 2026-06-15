CREATE TABLE IF NOT EXISTS api_keys (
    key_hash   TEXT        PRIMARY KEY,          -- SHA-256(raw_token), hex
    tenant_id  TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    active     BOOLEAN     NOT NULL DEFAULT TRUE
);
