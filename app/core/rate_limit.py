"""Rate limiter propio sobre Redis: sliding window counter via un script Lua
atomico (evita la condicion de carrera de leer/decidir/incrementar por
separado). No se uso slowapi (bloquea el event loop) ni fastapi-limiter
(mantenimiento esporadico)."""

import logging
import time
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import NoScriptError, RedisError

logger = logging.getLogger(__name__)

# KEYS[1] = clave de la ventana actual, KEYS[2] = clave de la ventana anterior
# ARGV[1] = tamano de ventana en segundos, ARGV[2] = limite, ARGV[3] = timestamp unix actual
_SLIDING_WINDOW_SCRIPT = """
local current_key = KEYS[1]
local previous_key = KEYS[2]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local current_count = tonumber(redis.call("GET", current_key) or "0")
local previous_count = tonumber(redis.call("GET", previous_key) or "0")

local elapsed = now % window
local weight = 1 - (elapsed / window)
local estimated = (previous_count * weight) + current_count

if estimated >= limit then
    return {0, tostring(estimated)}
end

redis.call("INCR", current_key)
redis.call("EXPIRE", current_key, window * 2)
return {1, tostring(estimated + 1)}
"""


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after_seconds: int


class RateLimiter:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._script_sha: str | None = None

    async def preload(self) -> None:
        self._script_sha = await self._redis.script_load(_SLIDING_WINDOW_SCRIPT)

    async def check(self, identity: str, limit: int, window_seconds: int) -> RateLimitResult:
        now = time.time()
        bucket = int(now // window_seconds)
        current_key = f"ratelimit:{identity}:{bucket}"
        previous_key = f"ratelimit:{identity}:{bucket - 1}"

        try:
            if self._script_sha is None:
                raise NoScriptError
            allowed, _ = await self._redis.evalsha(
                self._script_sha, 2, current_key, previous_key, window_seconds, limit, now
            )
        except NoScriptError:
            # Redis perdio el cache de scripts compilados (ej. tras un
            # restart del contenedor -- SCRIPT LOAD no persiste entre
            # reinicios del server): se manda el texto completo una vez mas
            # y se vuelve a cachear para las proximas llamadas.
            allowed, _ = await self._redis.eval(
                _SLIDING_WINDOW_SCRIPT, 2, current_key, previous_key, window_seconds, limit, now
            )
            self._script_sha = await self._redis.script_load(_SLIDING_WINDOW_SCRIPT)
        except (RedisError, ConnectionError, OSError) as exc:
            # Si Redis cae temporalmente, fail-open para mantener alta disponibilidad
            logger.warning("rate_limiter.redis_unavailable_fail_open", extra={"identity": identity, "error": str(exc)})
            return RateLimitResult(allowed=True, retry_after_seconds=0)

        return RateLimitResult(allowed=bool(int(allowed)), retry_after_seconds=0 if allowed else window_seconds)
