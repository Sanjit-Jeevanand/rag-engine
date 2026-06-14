from __future__ import annotations

import time

from redis.asyncio import Redis

_LUA = """
local key    = KEYS[1]
local cap    = tonumber(ARGV[1])
local now    = tonumber(ARGV[2])
local window = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count < cap then
    redis.call('ZADD', key, now, now .. '-' .. math.random(1000000))
    redis.call('EXPIRE', key, window)
    return 1
end
return 0
"""


async def check(
    redis: Redis,
    tenant_id: str,
    cap: int = 10,
    window_sec: int = 600,
) -> bool:
    key = f"ratelimit:{tenant_id}"
    now = time.time()
    result = await redis.eval(_LUA, 1, key, cap, now, window_sec)
    return bool(result)
