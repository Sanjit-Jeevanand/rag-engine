"""
Generate API keys and insert them into the Postgres api_keys table.

This script must be run from INSIDE the server (Postgres is not exposed
externally). The typical invocation is:

    ssh root@<IP> bash -s <<'EOF'
    docker compose -f /opt/rag-engine/infra/docker-compose.yml exec -T api \\
        python scripts/gen_api_keys.py \\
        --db-url postgresql://rag:rag@postgres:5432/rag \\
        --count 5 --prefix demo
    EOF

Output: prints the raw sk-* tokens to stdout once — they are NOT stored anywhere
else. Only a SHA-256 hash is written to Postgres.
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
import sys

import psycopg2


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def generate(db_url: str, count: int, prefix: str) -> list[tuple[str, str]]:
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()

    pairs: list[tuple[str, str]] = []
    for i in range(1, count + 1):
        raw = "sk-" + secrets.token_hex(24)
        tenant_id = f"{prefix}-{i:02d}"
        cur.execute(
            """
            INSERT INTO api_keys (key_hash, tenant_id, active)
            VALUES (%s, %s, TRUE)
            ON CONFLICT (key_hash) DO NOTHING
            """,
            (_hash(raw), tenant_id),
        )
        pairs.append((tenant_id, raw))

    conn.commit()
    cur.close()
    conn.close()
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate API keys into Postgres")
    parser.add_argument("--db-url", required=True, help="Postgres DSN")
    parser.add_argument("--count", type=int, default=5, help="Number of keys to create")
    parser.add_argument(
        "--prefix", default="user", help="Tenant ID prefix (e.g. 'demo')"
    )
    args = parser.parse_args()

    pairs = generate(args.db_url, args.count, args.prefix)

    print(
        f"\nGenerated {len(pairs)} key(s) — save these now, they will not be shown again:\n"
    )
    for tenant_id, raw in pairs:
        print(f"  {tenant_id:20s}  {raw}")
    print()


if __name__ == "__main__":
    try:
        main()
    except psycopg2.Error as exc:
        print(f"DB error: {exc}", file=sys.stderr)
        sys.exit(1)
