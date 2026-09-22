"""CLI de Ingesta Inicial y Población de Legislación Argentina (Bootstrap CLI).

Permite indexar leyes de forma selectiva o en lote desde el repositorio local legalize-ar,
poblando PostgreSQL (con búsqueda léxica FTS) y opcionalmente Pinecone (con embeddings densos).

Uso:
  # 1. Indexar leyes prioritarias con embeddings (CCC, Sociedades, Trabajo, CN, DNU 70/2023, etc.):
  python -m scripts.legal_bootstrap --priority

  # 2. Indexar leyes específicas:
  python -m scripts.legal_bootstrap --laws LEY-26994,DNU-70-2023

  # 3. Indexar solo en PostgreSQL (Modo Offline / Sin cuota de OpenAI):
  python -m scripts.legal_bootstrap --priority --skip-embeddings

  # 4. Ingesta masiva con límite:
  python -m scripts.legal_bootstrap --all --limit 50
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.models_orm import Base, LegalArticle, LegalLaw, LegalSyncRun
from app.services.legal.markdown_parser import ParsedLaw, parse_markdown_law
from app.services.rag_service import (
    build_embeddings_client,
    build_pinecone_client,
    embed_texts,
    ensure_index_exists,
)

logger = logging.getLogger("legal_bootstrap")

# Catálogo de leyes troncales del ordenamiento jurídico argentino y reformas de alto impacto
PRIORITY_LAWS = [
    # 1. Constitucional y Tratados
    "LEY-24430",    # Constitución de la Nación Argentina

    # 2. Civil, Comercial y Sociedades
    "LEY-26994",    # Código Civil y Comercial de la Nación (CCyC)
    "LEY-19550",    # Ley General de Sociedades
    "LEY-27349",    # Apoyo al Capital Emprendedor (SAS - Sociedades por Acciones Simplificadas)
    "LEY-24522",    # Ley de Concursos y Quiebras
    "LEY-25065",    # Ley de Tarjetas de Crédito
    "LEY-24452",    # Ley de Cheques
    "LEY-17801",    # Registro de la Propiedad Inmueble
    "LEY-22362",    # Ley de Marcas y Designaciones
    "LEY-11723",    # Régimen Legal de la Propiedad Intelectual
    "LEY-25506",    # Firma Digital y Documentos Electrónicos
    "LEY-17418",    # Ley de Seguros

    # 3. Laboral y Seguridad Social
    "LEY-27802",    # Ley de Modernización Laboral (2026 - Reforma LCT 245, plataformas, cese)
    "DEC-390-1976", # Ley de Contrato de Trabajo (LCT - T.O. Decreto 390/1976)
    "LEY-20744",    # Ley de Contrato de Trabajo (Sanción originaria 1974)
    "LEY-24557",    # Ley de Riesgos del Trabajo (ART)
    "LEY-24013",    # Ley Nacional de Empleo
    "LEY-25877",    # Régimen Laboral / Ordenamiento del Trabajo
    "LEY-23551",    # Asociaciones Sindicales y Tutela Sindical
    "LEY-14786",    # Conciliación Obligatoria en Conflictos de Trabajo
    "LEY-24635",    # SECLO (Conciliación Laboral Previa Obligatoria)
    "LEY-11544",    # Jornada de Trabajo Legal
    "LEY-26844",    # Régimen Especial de Casas Particulares
    "LEY-24241",    # Sistema Integrado de Jubilaciones y Pensiones

    # 4. Reformas Estructurales y Emergencia
    "DNU-70-2023",  # DNU Bases para la Reconstrucción de la Economía Argentina
    "LEY-27742",    # Ley de Bases y Puntos de Partida para la Libertad de los Argentinos
    "LEY-27743",    # Paquete Fiscal 2024 (Medidas Fiscales Paliativas, Blanqueo y Ganancias)
    "LEY-27737",    # Reforma de Alquileres de Octubre 2023 (Derecho Transitorio)
    "LEY-27551",    # Ley de Alquileres de 2020 (Norma derogada para derecho transitorio)

    # 5. Consumo, Datos y Competencia
    "LEY-24240",    # Ley de Defensa del Consumidor
    "LEY-27442",    # Ley de Defensa de la Competencia
    "LEY-25326",    # Protección de Datos Personales (Habeas Data)
    "LEY-27275",    # Acceso a la Información Pública

    # 6. Penal, Ciberdelito y Género
    "LEY-11179",    # Código Penal de la Nación
    "LEY-26388",    # Delitos Informáticos y Ciberdelitos
    "LEY-27401",    # Responsabilidad Penal de las Personas Jurídicas (Compliance)
    "LEY-26485",    # Protección Integral contra la Violencia hacia las Mujeres
    "LEY-24417",    # Protección contra la Violencia Familiar
    "LEY-26061",    # Protección Integral de Derechos de Niñas, Niños y Adolescentes
    "LEY-27610",    # Acceso a la Interrupción Voluntaria del Embarazo (IVE)
    "LEY-26529",    # Derechos del Paciente e Historia Clínica
    "LEY-24449",    # Ley Nacional de Tránsito y Seguridad Vial

    # 7. Procesal y Fiscal
    "LEY-17454",    # Código Procesal Civil y Comercial de la Nación (CPCCN)
    "LEY-23984",    # Código Procesal Penal de la Nación (CPPN)
    "LEY-11683",    # Procedimiento Tributario (ARCA / AFIP)
    "LEY-24977",    # Régimen Simplificado para Pequeños Contribuyentes (Monotributo)
]


async def ingest_single_law(
    relative_path: str,
    session: AsyncSession,
    settings: Any,
    embeddings_client: Any,
    pinecone_index: Any,
    skip_embeddings: bool,
    batch_size: int,
    force: bool = False,
) -> dict[str, int]:
    """Parsea una norma e inserta sus artículos en PostgreSQL y Pinecone con hashing selectivo."""
    full_path = os.path.join(settings.LEGALIZE_REPO_PATH, relative_path)
    if not os.path.exists(full_path):
        logger.warning("Archivo no encontrado: %s", full_path)
        return {"articles_indexed": 0, "articles_skipped": 0}

    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
        raw_md = f.read()

    fallback_id = os.path.basename(relative_path).replace(".md", "")
    parsed: ParsedLaw = parse_markdown_law(raw_md, fallback_id)

    # 1. Registrar o actualizar LegalLaw
    stmt_law = select(LegalLaw).where(LegalLaw.identifier == parsed.identifier)
    res_law = await session.execute(stmt_law)
    existing_law = res_law.scalar_one_or_none()

    if not existing_law:
        law_record = LegalLaw(
            identifier=parsed.identifier,
            title=parsed.title,
            country=parsed.country,
            rank=parsed.rank,
            status=parsed.status,
            publication_date=parsed.publication_date,
            last_updated=parsed.last_updated,
            enactment_date=parsed.enactment_date,
            department=parsed.department,
            source_url=parsed.source,
            infoleg_id=parsed.infoleg_id,
            reform_quality=parsed.reform_quality,
            law_metadata=parsed.extra_metadata,
        )
        session.add(law_record)
        await session.flush()
        law_id = law_record.id
        existing_hashes = {}
    else:
        existing_law.title = parsed.title
        existing_law.status = parsed.status
        existing_law.last_updated = parsed.last_updated
        existing_law.law_metadata = parsed.extra_metadata
        law_id = existing_law.id

        stmt_hashes = select(LegalArticle.article_number, LegalArticle.content_hash).where(
            LegalArticle.law_id == law_id
        )
        res_h = await session.execute(stmt_hashes)
        existing_hashes = {row[0]: row[1] for row in res_h.all()}

    articles_to_embed = []
    articles_to_save = []
    skipped_count = 0

    # 2. Comparar hashes de artículos
    for art in parsed.articles:
        prev_hash = existing_hashes.get(art.article_number)
        if not force and prev_hash == art.content_hash:
            skipped_count += 1
            continue

        p_id = f"{parsed.identifier}:{art.article_number}"
        articles_to_save.append((art, p_id))
        if not skip_embeddings and embeddings_client and pinecone_index:
            articles_to_embed.append((art, p_id))

    # 3. Generar embeddings por lotes si no se saltearon
    if articles_to_embed:
        texts = [a[0].contextualized_text for a in articles_to_embed]
        vectors_data = []
        embed_chunk_size = 50
        for b_idx in range(0, len(texts), embed_chunk_size):
            chunk = texts[b_idx : b_idx + embed_chunk_size]
            chunk_vectors = await embed_texts(chunk, embeddings_client)
            vectors_data.extend(chunk_vectors)

        pinecone_vectors = []
        for (art, p_id), vector in zip(articles_to_embed, vectors_data, strict=True):
            pinecone_vectors.append(
                {
                    "id": p_id,
                    "values": vector,
                    "metadata": {
                        "law_identifier": parsed.identifier,
                        "law_title": parsed.title,
                        "article_number": art.article_number,
                        "epigraph": art.epigraph or "",
                        "status": art.status,
                        "rank": parsed.rank,
                        "country": parsed.country,
                        "infoleg_url": parsed.source or "",
                        "text": art.content_raw[:1000],
                    },
                }
            )

        for b_start in range(0, len(pinecone_vectors), batch_size):
            batch = pinecone_vectors[b_start : b_start + batch_size]
            await pinecone_index.upsert(vectors=batch, namespace=settings.PINECONE_LEGAL_NAMESPACE)

    # 4. Actualizar PostgreSQL
    for art, p_id in articles_to_save:
        await session.execute(
            delete(LegalArticle).where(
                LegalArticle.law_id == law_id,
                LegalArticle.article_number == art.article_number,
            )
        )
        art_record = LegalArticle(
            law_id=law_id,
            law_identifier=parsed.identifier,
            article_number=art.article_number,
            article_order=art.article_order,
            epigraph=art.epigraph,
            content=art.content_raw,
            text_searchable=art.contextualized_text,
            book=art.hierarchy.book,
            title_section=art.hierarchy.title_section,
            chapter=art.hierarchy.chapter,
            section=art.hierarchy.section,
            status=art.status,
            content_hash=art.content_hash,
            pinecone_id=p_id,
            article_metadata={
                "incisos": art.incisos,
                "editorial_notes": art.editorial_notes,
            },
        )
        session.add(art_record)

    await session.commit()
    return {"articles_indexed": len(articles_to_save), "articles_skipped": skipped_count}


async def run_bootstrap(args: argparse.Namespace) -> None:
    settings = get_settings()
    configure_logging(settings)

    print("=================================================================")
    print("      BOOTSTRAP DE LEGISLACIÓN NACIONAL (legalize-ar)           ")
    print("=================================================================")
    print(f"Repositorio local: {settings.LEGALIZE_REPO_PATH}")
    print(f"Modo Embeddings: {'DESHABILITADO (--skip-embeddings)' if args.skip_embeddings else 'HABILITADO'}")

    # Determinar qué leyes indexar
    files_to_process: list[str] = []
    country_code = getattr(settings, "LEGALIZE_COUNTRY_CODE", "ar")
    country_dir = os.path.join(settings.LEGALIZE_REPO_PATH, country_code)
    target_dir_name = country_code
    if not os.path.exists(country_dir) and os.path.exists(os.path.join(settings.LEGALIZE_REPO_PATH, "ar")):
        country_dir = os.path.join(settings.LEGALIZE_REPO_PATH, "ar")
        target_dir_name = "ar"

    if not os.path.exists(country_dir):
        print(f"ERROR: No se encontró la carpeta '{country_code}' en {settings.LEGALIZE_REPO_PATH}")
        sys.exit(1)

    if args.laws:
        specified = [l.strip().upper() for l in args.laws.split(",") if l.strip()]
        for law_id in specified:
            fname = f"{law_id}.md"
            if os.path.exists(os.path.join(country_dir, fname)):
                files_to_process.append(f"{target_dir_name}/{fname}")
            else:
                print(f"[!] Aviso: No se encontró el archivo {target_dir_name}/{fname}")

    elif args.priority:
        print(f"\n-> Seleccionando catálogo de leyes prioritarias ({settings.LEGALIZE_COUNTRY_NAME})...")
        for law_id in PRIORITY_LAWS:
            fname = f"{law_id}.md"
            if os.path.exists(os.path.join(country_dir, fname)):
                files_to_process.append(f"{target_dir_name}/{fname}")
            else:
                print(f"[!] Ley prioritaria no hallada: {target_dir_name}/{fname}")

    elif args.all:
        print(f"\n-> Modo completo: escaneando todas las normas en {target_dir_name}/...")
        all_mds = sorted([f for f in os.listdir(country_dir) if f.endswith(".md")])
        if args.limit:
            all_mds = all_mds[: args.limit]
        files_to_process = [f"{target_dir_name}/{f}" for f in all_mds]

    else:
        print("\nDebe especificar una modalidad: --priority, --laws LEY-1234, o --all")
        sys.exit(1)

    print(f"Total de normas a procesar: {len(files_to_process)}")
    print("-----------------------------------------------------------------")

    engine = create_async_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    embeddings_client = None
    pinecone_index = None

    if not args.skip_embeddings:
        try:
            embeddings_client = build_embeddings_client(settings)
            pinecone_c = build_pinecone_client(settings)
            await ensure_index_exists(pinecone_c, settings)
            pinecone_index = await pinecone_c.index(name=settings.PINECONE_INDEX_NAME)
        except Exception as exc:
            print(f"[!] Error inicializando clientes de vectores: {exc}")
            print("    Continuando con --skip-embeddings forzado (solo PostgreSQL FTS)...")
            args.skip_embeddings = True

    start_total = time.time()
    total_articles = 0
    total_skipped = 0

    for i, rel_path in enumerate(files_to_process, 1):
        law_id = os.path.basename(rel_path).replace(".md", "")
        t0 = time.time()
        try:
            async with sessionmaker() as session:
                counts = await ingest_single_law(
                    relative_path=rel_path,
                    session=session,
                    settings=settings,
                    embeddings_client=embeddings_client,
                    pinecone_index=pinecone_index,
                    skip_embeddings=args.skip_embeddings,
                    batch_size=args.batch_size,
                    force=args.force,
                )
            dt = time.time() - t0
            total_articles += counts["articles_indexed"]
            total_skipped += counts["articles_skipped"]
            print(
                f"[{i:02d}/{len(files_to_process):02d}] {law_id:<15} "
                f"-> Artículos: {counts['articles_indexed']:4d} indexados, "
                f"{counts['articles_skipped']:4d} omitidos ({dt:.2f}s)"
            )
        except Exception as exc:
            print(f"[{i:02d}/{len(files_to_process):02d}] {law_id:<15} -> ERROR: {exc}")

    elapsed = time.time() - start_total
    print("=================================================================")
    print("                    RESUMEN DE INGESTA                           ")
    print("=================================================================")
    print(f"Normas procesadas:        {len(files_to_process)}")
    print(f"Artículos indexados:      {total_articles}")
    print(f"Artículos omitidos (hash):{total_skipped}")
    print(f"Tiempo total:             {elapsed:.2f}s")
    print("=================================================================")

    # Registrar LegalSyncRun
    try:
        async with sessionmaker() as session:
            sync_rec = LegalSyncRun(
                sync_source="bootstrap_cli",
                files_added=len(files_to_process),
                articles_indexed=total_articles,
                articles_skipped_unchanged=total_skipped,
                duration_seconds=elapsed,
                status="COMPLETED",
            )
            session.add(sync_rec)
            await session.commit()
    except Exception:
        pass

    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap de Legislación Argentina en PostgreSQL + Pinecone")
    parser.add_argument("--priority", action="store_true", help="Indexar leyes y códigos prioritarios de Argentina")
    parser.add_argument("--laws", type=str, help="Lista de identificadores separados por coma (ej: LEY-26994,DNU-70-2023)")
    parser.add_argument("--all", action="store_true", help="Indexar todas las leyes de repo_legalize_ar/ar/")
    parser.add_argument("--limit", type=int, help="Límite máximo de leyes a procesar con --all")
    parser.add_argument("--skip-embeddings", action="store_true", help="Omitir generación de vectores OpenAI (solo PostgreSQL FTS)")
    parser.add_argument("--force", action="store_true", help="Forzar reindexación y regeneración de embeddings ignorando hashes existentes")
    parser.add_argument("--batch-size", type=int, default=100, help="Tamaño de lote para upsert en Pinecone")

    args = parser.parse_args()
    asyncio.run(run_bootstrap(args))


if __name__ == "__main__":
    main()



