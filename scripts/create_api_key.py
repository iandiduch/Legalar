"""Script CLI para generar API keys de cliente o administrador.

Uso:
    python -m scripts.create_api_key --name "Frontend Web" --scope client
    python -m scripts.create_api_key --name "Admin Console" --scope admin
"""

import argparse
import asyncio
import sys

from app.core.config import get_settings
from app.db.session import build_engine, build_sessionmaker
from app.domain.models import ApiKeyScope
from app.services.api_key_service import ApiKeyService


async def generate_key(name: str, scope: ApiKeyScope) -> None:
    settings = get_settings()
    engine = build_engine(settings)
    sessionmaker = build_sessionmaker(engine)

    try:
        async with sessionmaker() as session:
            service = ApiKeyService(session, settings)
            created = await service.create_key(
                name=name,
                scope=scope,
                created_by="cli.create_api_key",
            )
            await session.commit()
    except Exception as exc:
        print(f"\n❌ Error al generar la API key en la base de datos: {exc}")
        print("Asegurate de que PostgreSQL esté corriendo y las tablas inicializadas (`python -m scripts.init_db`).\n")
        await engine.dispose()
        sys.exit(1)

    await engine.dispose()

    print("\n" + "=" * 65)
    print("  🔑 API KEY GENERADA EXITOSAMENTE")
    print("=" * 65)
    print(f"  Nombre:        {created.name}")
    print(f"  Key ID:        {created.key_id}")
    print(f"  Scope:         {created.scope}")
    print(f"  API Key (Raw): {created.plaintext_key}")
    print("=" * 65)
    print("  ⚠️  IMPORTANTE: Guardá esta clave en tu archivo `.env`")
    print("  como VITE_API_KEY en el frontend o en tu cliente HTTP.")
    print("  No volverá a mostrarse en texto plano.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar API Key para Legalar")
    parser.add_argument("--name", type=str, default="frontend-app", help="Nombre descriptivo de la clave")
    parser.add_argument(
        "--scope",
        type=str,
        choices=["client", "admin"],
        default="client",
        help="Nivel de permisos ('client' para consultas/frontend, 'admin' para gestión)",
    )
    args = parser.parse_args()

    asyncio.run(generate_key(args.name, ApiKeyScope(args.scope)))
