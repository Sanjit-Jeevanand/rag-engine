"""
Generate API keys and insert their SHA-256 hashes into Postgres.

Prints raw tokens to stdout — store them securely, they are not saved anywhere.

Usage:
    uv run python scripts/gen_api_keys.py --db-url postgresql://rag:rag@localhost:5432/rag
    uv run python scripts/gen_api_keys.py --db-url $RAG_DB_URL --count 5 --prefix acme
"""

import argparse
import asyncio
import hashlib
import secrets


async def run(db_url: str, count: int, prefix: str) -> None:
    import asyncpg

    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=1)

    print(f"Generating {count} key(s)...\n")
    for i in range(count):
        raw = f"sk-{secrets.token_hex(24)}"
        key_hash = hashlib.sha256(raw.encode()).hexdigest()
        tenant_id = f"{prefix}-{i + 1:02d}" if count > 1 else prefix

        await pool.execute(
            "INSERT INTO api_keys (key_hash, tenant_id) VALUES ($1, $2)"
            " ON CONFLICT (key_hash) DO NOTHING",
            key_hash,
            tenant_id,
        )
        print(f"  tenant: {tenant_id}")
        print(f"  token:  {raw}")
        print()

    await pool.close()
    print("Done. Share the tokens above — hashes only are stored in Postgres.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", required=True, help="Postgres DSN")
    parser.add_argument(
        "--count", type=int, default=10, help="Number of keys (default: 10)"
    )
    parser.add_argument(
        "--prefix", default="user", help="Tenant ID prefix (default: user)"
    )
    args = parser.parse_args()

    asyncio.run(run(args.db_url, args.count, args.prefix))


if __name__ == "__main__":
    main()
