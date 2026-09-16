from redis.asyncio import Redis

from app.core.config import Settings


def build_redis_client(settings: Settings, *, socket_timeout: float | None = 15.0) -> Redis:
    return Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=socket_timeout)
