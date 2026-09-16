"""Pool psycopg dedicado exclusivamente al checkpointer de LangGraph. Nunca se
comparte con el engine SQLAlchemy/asyncpg de app/db/session.py: son dos dueños
distintos (LangGraph gestiona su propio schema internamente, nosotros el nuestro).

`.setup()` corre una unica vez desde scripts/init_db.py -- ver ese archivo.
"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import Settings
from app.domain.models import (
    ConfidenceLevel,
    LegalRank,
    LegalStatus,
    QueryIntent,
)
from app.schemas.legal.chat import LegalValidationSummary
from app.schemas.legal.citation import LegalCitation
from app.schemas.legal.diff import ArticleDiff, DiffResponse

_CONNECTION_KWARGS = {"autocommit": True, "row_factory": dict_row}
_ALLOWED_CHECKPOINT_TYPES = [
    LegalCitation,
    LegalValidationSummary,
    DiffResponse,
    ArticleDiff,
    ConfidenceLevel,
    QueryIntent,
    LegalRank,
    LegalStatus,
]


async def build_checkpointer_pool(settings: Settings, timeout: float = 5.0) -> AsyncConnectionPool:
    pool = AsyncConnectionPool(
        conninfo=settings.postgres_dsn,
        max_size=settings.CHECKPOINTER_POOL_MAX_SIZE,
        min_size=settings.CHECKPOINTER_POOL_MIN_SIZE,
        timeout=timeout,
        kwargs=_CONNECTION_KWARGS,
        open=False,
    )
    await pool.open(wait=False)
    return pool


def build_checkpointer(pool: AsyncConnectionPool) -> AsyncPostgresSaver:
    serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_CHECKPOINT_TYPES)
    return AsyncPostgresSaver(pool, serde=serde)
