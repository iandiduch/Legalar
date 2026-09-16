"""Tests de la capa de seguridad: hashing, jerarquia de scopes, allowlist de
IP (sin infraestructura) y el rate limiter (requiere Redis real -- mismo
criterio que los tests de Postgres: `docker compose up -d redis` antes de
correrlo)."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.core.exceptions import AuthorizationError
from app.core.ip_filter import is_ip_in_allowlist
from app.core.rate_limit import RateLimiter
from app.core.security import generate_api_key, hash_api_key, require_scope
from app.db.models_orm import ApiKey
from app.domain.models import ApiKeyScope


def _settings_with_pepper(pepper: str, allowlist: str = ""):
    return MagicMock(API_KEY_PEPPER=MagicMock(get_secret_value=lambda: pepper), ADMIN_IP_ALLOWLIST=allowlist)


def test_generate_api_key_has_expected_shape():
    key = generate_api_key()
    assert key.startswith("isk_")
    assert len(key) > 20  # "isk_" + token_urlsafe(32) es bastante mas largo que esto


def test_hash_api_key_is_deterministic_with_same_pepper():
    settings = _settings_with_pepper("mismo-pepper")
    plaintext = generate_api_key()
    assert hash_api_key(plaintext, settings) == hash_api_key(plaintext, settings)


def test_hash_api_key_differs_for_different_plaintext():
    settings = _settings_with_pepper("mismo-pepper")
    assert hash_api_key(generate_api_key(), settings) != hash_api_key(generate_api_key(), settings)


def test_hash_api_key_differs_with_different_pepper():
    plaintext = generate_api_key()
    hash_a = hash_api_key(plaintext, _settings_with_pepper("pepper-a"))
    hash_b = hash_api_key(plaintext, _settings_with_pepper("pepper-b"))
    assert hash_a != hash_b


async def test_require_scope_admin_key_passes_admin_and_client_checks():
    # settings se pasa directo (require_scope lo recibe por Depends) -- el chequeo
    # ADMIN lo usa para ADMIN_IP_ALLOWLIST, a diferencia del caso CLIENT-pide-ADMIN
    # de abajo, que nunca llega tan lejos porque la jerarquia de scopes ya corta antes.
    settings = _settings_with_pepper("irrelevante", allowlist="")
    admin_key = ApiKey(key_id=uuid4(), scope=ApiKeyScope.ADMIN.value, name="test-admin")
    fake_request = MagicMock()

    result_client = await require_scope(ApiKeyScope.CLIENT)(
        request=fake_request, key_record=admin_key, settings=settings
    )
    result_admin = await require_scope(ApiKeyScope.ADMIN)(request=fake_request, key_record=admin_key, settings=settings)

    assert result_client is admin_key
    assert result_admin is admin_key


async def test_require_scope_client_key_fails_admin_check():
    # settings=None: la jerarquia de scopes corta antes de que require_scope
    # llegue a tocar settings en el caso ADMIN, asi que un valor real no hace falta.
    client_key = ApiKey(key_id=uuid4(), scope=ApiKeyScope.CLIENT.value, name="test-client")
    fake_request = MagicMock()

    result = await require_scope(ApiKeyScope.CLIENT)(request=fake_request, key_record=client_key, settings=None)
    assert result is client_key

    with pytest.raises(AuthorizationError):
        await require_scope(ApiKeyScope.ADMIN)(request=fake_request, key_record=client_key, settings=None)


def test_is_ip_in_allowlist_empty_allows_everything():
    assert is_ip_in_allowlist("203.0.113.5", "") is True


def test_is_ip_in_allowlist_matches_cidr():
    assert is_ip_in_allowlist("10.20.5.7", "10.20.0.0/16, 192.168.1.0/24") is True


def test_is_ip_in_allowlist_rejects_outside_cidr():
    assert is_ip_in_allowlist("203.0.113.5", "10.20.0.0/16") is False


def test_is_ip_in_allowlist_rejects_malformed_ip():
    assert is_ip_in_allowlist("no-es-una-ip", "10.20.0.0/16") is False


@pytest.mark.asyncio
async def test_rate_limiter_allows_under_limit_and_blocks_over(test_settings):
    from redis.asyncio import Redis

    redis = Redis.from_url(test_settings.redis_url, decode_responses=False)
    try:
        await redis.ping()
    except Exception:  # noqa: BLE001 - Salta test si Redis local no está disponible
        pytest.skip("Redis no disponible -- correr 'docker compose up -d redis' para este test")

    limiter = RateLimiter(redis)
    await limiter.preload()
    identity = f"test:{uuid4()}"

    for _ in range(3):
        result = await limiter.check(identity, limit=3, window_seconds=60)
        assert result.allowed is True

    blocked = await limiter.check(identity, limit=3, window_seconds=60)
    assert blocked.allowed is False
    assert blocked.retry_after_seconds == 60

    await redis.aclose()


@pytest.mark.asyncio
async def test_rate_limiter_fail_open_on_redis_error():
    from unittest.mock import AsyncMock

    from redis.exceptions import ConnectionError as RedisConnError

    mock_redis = AsyncMock()
    mock_redis.evalsha = AsyncMock(side_effect=RedisConnError("Redis down"))
    limiter = RateLimiter(mock_redis)
    limiter._script_sha = "fake_sha"

    result = await limiter.check("test-identity", limit=10, window_seconds=60)
    assert result.allowed is True
    assert result.retry_after_seconds == 0

