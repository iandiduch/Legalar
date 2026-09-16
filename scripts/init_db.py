"""Setup de infraestructura y base de datos para el Agente Legal Argentino.

Se ejecuta UNA sola vez antes de levantar el API y los workers:
  1. Crea las tablas propias de legislación nacional (legal_laws, legal_articles,
     legal_revisions, legal_sync_runs), jobs de ingesta documental, api_keys y prompts.
  2. Inicializa las tablas del checkpointer de LangGraph (AsyncPostgresSaver.setup()).
  3. Asegura el índice de Pinecone si la API key está configurada.
  4. Siembra los prompts default de los agentes legales si la tabla agent_prompts está vacía.
  5. Siembra la primera API key admin (BOOTSTRAP_ADMIN_API_KEY) si api_keys está vacía.

Uso: python -m scripts.init_db
"""

import asyncio
import logging
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.checkpointer import build_checkpointer, build_checkpointer_pool
from app.db.models_orm import Base
from app.services.api_key_service import seed_bootstrap_admin_key
from app.services.prompt_manager import PromptManager
from app.services.rag_service import build_pinecone_client, ensure_index_exists

logger = logging.getLogger(__name__)

_DEFAULTS_DIR = Path(__file__).resolve().parent.parent / "app" / "agents" / "prompts" / "defaults"


async def main() -> None:
    settings = get_settings()
    configure_logging(settings)

    logger.info("init_db.creating_app_tables")
    engine = create_async_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if engine.dialect.name == "postgresql":
            await conn.execute(
                text(
                    """
                    DO $$
                    BEGIN
                        -- Tablas legales
                        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'legal_laws') THEN
                            ALTER TABLE legal_laws
                                ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC',
                                ALTER COLUMN updated_at TYPE TIMESTAMPTZ USING updated_at AT TIME ZONE 'UTC';
                        END IF;
                        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'legal_articles') THEN
                            ALTER TABLE legal_articles
                                ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC',
                                ALTER COLUMN updated_at TYPE TIMESTAMPTZ USING updated_at AT TIME ZONE 'UTC';
                        END IF;
                        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'legal_revisions') THEN
                            ALTER TABLE legal_revisions
                                ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC';
                        END IF;
                        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'legal_sync_runs') THEN
                            ALTER TABLE legal_sync_runs
                                ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC';
                        END IF;

                        -- Tablas operativas
                        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'ingestion_jobs') THEN
                            ALTER TABLE ingestion_jobs
                                ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC',
                                ALTER COLUMN started_at TYPE TIMESTAMPTZ USING started_at AT TIME ZONE 'UTC',
                                ALTER COLUMN completed_at TYPE TIMESTAMPTZ USING completed_at AT TIME ZONE 'UTC',
                                ALTER COLUMN updated_at TYPE TIMESTAMPTZ USING updated_at AT TIME ZONE 'UTC';
                        END IF;
                        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'agent_prompts') THEN
                            ALTER TABLE agent_prompts
                                ALTER COLUMN updated_at TYPE TIMESTAMPTZ USING updated_at AT TIME ZONE 'UTC';
                        END IF;
                        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'api_keys') THEN
                            ALTER TABLE api_keys
                                ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC',
                                ALTER COLUMN last_used_at TYPE TIMESTAMPTZ USING last_used_at AT TIME ZONE 'UTC',
                                ALTER COLUMN revoked_at TYPE TIMESTAMPTZ USING revoked_at AT TIME ZONE 'UTC';
                        END IF;
                        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'document_chunks') THEN
                            ALTER TABLE document_chunks
                                ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC';
                        END IF;
                    EXCEPTION WHEN OTHERS THEN
                        NULL;
                    END $$;
                    """
                )
            )

    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    logger.info("init_db.seeding_prompts")
    if _DEFAULTS_DIR.exists():
        await PromptManager(sessionmaker).seed_defaults_if_empty(_DEFAULTS_DIR)

    logger.info("init_db.seeding_bootstrap_admin_key")
    await seed_bootstrap_admin_key(sessionmaker, settings)

    await engine.dispose()

    logger.info("init_db.checkpointer_setup")
    try:
        pool = await build_checkpointer_pool(settings)
        try:
            await build_checkpointer(pool).setup()
        finally:
            await pool.close()
    except Exception as exc:
        logger.warning("No se pudo inicializar checkpointer de LangGraph (¿PostgreSQL offline?): %s", exc)

    logger.info("init_db.ensure_pinecone_index")
    if settings.PINECONE_API_KEY and settings.PINECONE_API_KEY.get_secret_value():
        try:
            pinecone_client = build_pinecone_client(settings)
            try:
                await ensure_index_exists(pinecone_client, settings)
            finally:
                await pinecone_client.close()
        except Exception as exc:
            logger.warning("No se pudo conectar con Pinecone (se usará modo local/PostgreSQL): %s", exc)
    else:
        logger.info("PINECONE_API_KEY no configurada. Saltando verificación de índice Pinecone.")

    logger.info("init_db.done")


if __name__ == "__main__":
    asyncio.run(main())


